import torch
from transformers import AutoTokenizer, AutoModel, GPT2Tokenizer, GPT2LMHeadModel, T5Tokenizer, T5ForConditionalGeneration
import faiss
import numpy as np
import gensim
from gensim import corpora
import spacy
import pandas as pd
import wikipediaapi
import os
import pickle  

# Initialize retriever and generator models
retriever_model_name = "bert-base-uncased"
retriever_tokenizer = AutoTokenizer.from_pretrained(retriever_model_name)
retriever_model = AutoModel.from_pretrained(retriever_model_name)

generator_model_name = "gpt2"
generator_tokenizer = GPT2Tokenizer.from_pretrained(generator_model_name)
generator_model = GPT2LMHeadModel.from_pretrained(generator_model_name)

# Initialize summarizer model and tokenizer
summarizer_model_name = "t5-small"  # Or use "t5-base", "t5-large"
summarizer_tokenizer = T5Tokenizer.from_pretrained(summarizer_model_name)
summarizer_model = T5ForConditionalGeneration.from_pretrained(summarizer_model_name)

#Load your dataset
df = pd.read_csv('../dataset/combined_dataset_training.csv', encoding='latin1')
print(df.head())

# Extract countries and policy descriptions
countries = df['country'].tolist()
documents = df['policy_description'].tolist()

# Combine countries and policy descriptions
combined_documents = [f"{country}: {policy}" for country, policy in zip(countries, documents)]

# Preprocessing
def preprocess_text(texts):
    return [[word for word in doc.lower().split() if word.isalnum()] for doc in texts]

preprocessed_docs = preprocess_text(combined_documents)
dictionary = corpora.Dictionary(preprocessed_docs)
corpus = [dictionary.doc2bow(doc) for doc in preprocessed_docs]

# Entity Recognition
nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

document_entities = [extract_entities(doc) for doc in combined_documents]

# Topic Modeling
lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=2, id2word=dictionary, passes=15)

def get_topic_distribution(text):
    bow = dictionary.doc2bow(preprocess_text([text])[0])
    return dict(lda_model.get_document_topics(bow))

document_topics = [get_topic_distribution(doc) for doc in combined_documents]

def summarize_documents(docs, max_length=150):
    summaries = []
    for doc in docs:
        input_text = "summarize: " + doc
        inputs = summarizer_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = summarizer_model.generate(inputs, max_length=max_length, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
    return summaries
    
def contextual_summarization(query, document, max_length=150):
    input_text = f"summarize: {query} context: {document}"
    inputs = summarizer_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = summarizer_model.generate(inputs, max_length=max_length, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Encode documents using the retriever model
document_embeddings = []
for doc in combined_documents:
    inputs = retriever_tokenizer(doc, return_tensors="pt", truncation=True, padding=True)
    outputs = retriever_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    document_embeddings.append(embeddings.detach().numpy())

document_embeddings = np.vstack(document_embeddings)

# Create a FAISS index and add document embeddings
index = faiss.IndexFlatL2(document_embeddings.shape[1])
index.add(document_embeddings)

# Create directory if it does not exist
os.makedirs('saved_data', exist_ok=True)

# Save the FAISS index to a file in the 'saved_data' directory
faiss.write_index(index, 'saved_data/retriever_index.faiss')
    
# Save the document embeddings to a NumPy file in the 'saved_data' directory
with open('saved_data/document_embeddings.npy', 'wb') as f:
    np.save(f, document_embeddings)
    
# Save metadata including combined documents, document entities, document topics, dictionary, and LDA model in the 'saved_data' directory
with open('saved_data/metadata.pkl', 'wb') as f:
    pickle.dump({
        'combined_documents': combined_documents,
        'document_entities': document_entities,
        'document_topics': document_topics,
        'dictionary': dictionary,
        'lda_model': lda_model
    }, f)

def retrieve_documents(query, countries=None, top_k=4):
    inputs = retriever_tokenizer(query, return_tensors="pt", truncation=True, padding=True)
    outputs = retriever_model(**inputs)
    query_embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    
    # Perform FAISS search
    distances, indices = index.search(query_embedding, len(combined_documents))
    
    # Get topics and entities of the query
    query_topics = get_topic_distribution(query)
    query_entities = extract_entities(query)
    
    # Define key terms to boost relevance (e.g., "tax", "incentive")
    key_terms = {"tax", "incentive", "EV", "electric vehicle"}

    # Score documents based on topic and entity relevance
    doc_scores = []
    for i in range(len(indices[0])):
        idx = indices[0][i]

        # Prioritize based on countries if provided
        if countries and not any(country.lower() in combined_documents[idx].lower() for country in countries):
            continue

        doc_topic_score = sum(query_topics.get(topic, 0) * weight for topic, weight in document_topics[idx].items())
        doc_entity_score = len(set(ent[0] for ent in query_entities) & set(ent[0] for ent in document_entities[idx]))

        # Keyword matching score
        doc_text = combined_documents[idx].lower()
        keyword_score = sum(1 for term in key_terms if term in doc_text)

        # Combine scores with appropriate weights
        total_score = 0.5 * distances[0][i] - 2.0 * doc_topic_score + 2.0 * doc_entity_score - 3.0 * keyword_score

        # Boost score if the document matches any of the countries
        if any(country.lower() in combined_documents[idx].lower() for country in countries):
            total_score *= 0.5

        doc_scores.append((total_score, idx))

    # Sort documents by the combined score
    doc_scores.sort(key=lambda x: x[0])

    # Print scores and corresponding documents
    print("Scores and Retrieved Documents:")
    for score, idx in doc_scores[:top_k]:
        print(f"Score: {score}, Document: {combined_documents[idx]}")

    return [combined_documents[i] for _, i in doc_scores[:top_k]]

def generate_response(query, countries=None):
    retrieved_docs = retrieve_documents(query, countries)
    summaries = summarize_documents(retrieved_docs)
    context = " ".join(summaries)
    
    # Ensure context length does not exceed the model's max length
    max_input_length = generator_tokenizer.model_max_length - 20  # Reserve tokens for question and answer parts
    context = generator_tokenizer.decode(generator_tokenizer.encode(context, max_length=max_input_length, truncation=True))

    # Augment the query with country-specific context
    if countries:
        country_list = ", ".join(countries)
        augmented_query = f"Question: {query}\n\nContext: {context}\n\nAnswer:"
    else:
        augmented_query = f"Question: {query}\n\nContext: {context}\n\nAnswer:"
    
    inputs = generator_tokenizer.encode(augmented_query, return_tensors="pt")
    
    outputs = generator_model.generate(
        inputs, 
        max_length=1000, 
        temperature=0.7, 
        top_k=50, 
        top_p=0.95, 
        repetition_penalty=1.2  # Penalize repetition
    )
    
    response = generator_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Example usage
query = "What tax incentives does Norway provide for electric vehicles?"
# country = ["United Kingdom"]
country = ["Norway"]
response = generate_response(query, country)
print("Response:")
print(response)