import torch
from transformers import AutoTokenizer, AutoModel, GPT2Tokenizer, GPT2LMHeadModel
import faiss
import numpy as np
import gensim
from gensim import corpora
import spacy

# Load spaCy model for NER
nlp = spacy.load("en_core_web_sm")

# Initialize retriever and generator models
retriever_model_name = "bert-base-uncased"
retriever_tokenizer = AutoTokenizer.from_pretrained(retriever_model_name)
retriever_model = AutoModel.from_pretrained(retriever_model_name)

generator_model_name = "gpt2"
generator_tokenizer = GPT2Tokenizer.from_pretrained(generator_model_name)
generator_model = GPT2LMHeadModel.from_pretrained(generator_model_name)

# Example documents
documents = [
    "The capital of France is Paris.",
    "The moon orbits the Earth.",
    "The largest ocean on Earth is the Pacific Ocean."
]

# Topic Modeling
def preprocess_text(texts):
    return [[word for word in doc.lower().split() if word.isalnum()] for doc in texts]

preprocessed_docs = preprocess_text(documents)
dictionary = corpora.Dictionary(preprocessed_docs)
corpus = [dictionary.doc2bow(doc) for doc in preprocessed_docs]
lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=2, id2word=dictionary, passes=15)

def get_topic_distribution(text):
    bow = dictionary.doc2bow(preprocess_text([text])[0])
    return dict(lda_model.get_document_topics(bow))

document_topics = [get_topic_distribution(doc) for doc in documents]

# Entity Recognition
def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

document_entities = [extract_entities(doc) for doc in documents]

# Encode documents using the retriever model
document_embeddings = []
for doc in documents:
    inputs = retriever_tokenizer(doc, return_tensors="pt", truncation=True, padding=True)
    outputs = retriever_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    document_embeddings.append(embeddings.detach().numpy())

document_embeddings = np.vstack(document_embeddings)
# Create a FAISS index and add document embeddings
index = faiss.IndexFlatL2(document_embeddings.shape[1])
index.add(document_embeddings)

def retrieve_documents(query, top_k=2):
    inputs = retriever_tokenizer(query, return_tensors="pt", truncation=True, padding=True)
    outputs = retriever_model(**inputs)
    query_embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    
    # Perform FAISS search
    distances, indices = index.search(query_embedding, min(top_k, len(documents)))
    
    # Debugging output
    print(f"indices: {indices}")
    print(f"documents length: {len(documents)}")
    
    # Get topics and entities of the query
    query_topics = get_topic_distribution(query)
    query_entities = extract_entities(query)
    
    # Score documents based on topic and entity relevance
    doc_scores = []
    for idx in range(min(top_k, len(documents))):
        if indices[0][idx] < len(documents):  # Check if index is within bounds
            doc_index = indices[0][idx]
            doc_topic_score = sum(query_topics.get(topic, 0) * weight for topic, weight in document_topics[doc_index].items())
            doc_entity_score = len(set(ent[0] for ent in query_entities) & set(ent[0] for ent in document_entities[doc_index]))
            total_score = distances[0][idx] - doc_topic_score + doc_entity_score
            doc_scores.append((total_score, doc_index))
        else:
            print(f"Index {indices[0][idx]} is out of bounds for documents list")
    
    # Sort documents by the combined score
    doc_scores.sort(key=lambda x: x[0])
    return [documents[i] for _, i in doc_scores[:top_k]]


def generate_response(query):
    retrieved_docs = retrieve_documents(query)
    context = " ".join(retrieved_docs)
    
    # Ensure context length does not exceed the model's max length
    max_input_length = generator_tokenizer.model_max_length - 20  # Reserve tokens for question and answer parts
    context = generator_tokenizer.decode(generator_tokenizer.encode(context, max_length=max_input_length, truncation=True))
    
    augmented_query = f"Question: {query}\n\nContext: {context}\n\nAnswer:"
    inputs = generator_tokenizer.encode(augmented_query, return_tensors="pt")
    
    outputs = generator_model.generate(
        inputs, 
        max_length=150, 
        temperature=0.7, 
        top_k=50, 
        top_p=0.95, 
        repetition_penalty=1.2  # Penalize repetition
    )
    
    response = generator_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Example usage
query = "What is the largest ocean?"
response = generate_response(query)
print(response)
