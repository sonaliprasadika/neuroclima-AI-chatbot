import string
import pandas as pd
import numpy as np
import pickle
import faiss
from transformers import (
    BartForConditionalGeneration, BartTokenizer,
    DistilBertForQuestionAnswering, DistilBertTokenizer,
    T5ForConditionalGeneration, T5Tokenizer,
    GPT2LMHeadModel, GPT2Tokenizer,
    pipeline, AutoTokenizer, AutoModel
)
from langchain.chains import LLMChain
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
import spacy
from gensim import corpora
from gensim.models import LdaModel
import logging

# Configure logging
logging.basicConfig(filename='bulk_index_errors.log', level=logging.ERROR)

# Load the indexed data
faiss_index = faiss.read_index('saved_data/retriever_index.faiss')
document_embeddings = np.load('saved_data/document_embeddings.npy')

with open('saved_data/metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

combined_documents = metadata['combined_documents']
document_entities = metadata['document_entities']
document_topics = metadata['document_topics']
dictionary = metadata['dictionary']
lda_model = metadata['lda_model']

# Initialize spaCy model for entity recognition
nlp = spacy.load("en_core_web_sm")

# Define a function for entity recognition
def entity_recognition(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

# Define a function for topic modeling
def topic_modeling(docs, num_topics=3, num_words=4):
    stopwords = set(nlp.Defaults.stop_words)
    texts = []
    for doc in docs:
        tokens = doc.lower().split()
        tokens = [word.strip(string.punctuation) for word in tokens if word not in stopwords and len(word) > 1]
        texts.append(tokens)
        
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
    
    topics = []
    for i, topic in lda.show_topics(formatted=False, num_words=num_words):
        topics.append([word for word, _ in topic if word not in stopwords and len(word) > 1])
    
    return topics

# Initialize retriever model and tokenizer
retriever_model_name = "bert-base-uncased"
retriever_tokenizer = AutoTokenizer.from_pretrained(retriever_model_name)
retriever_model = AutoModel.from_pretrained(retriever_model_name)

# Search documents using FAISS and the stored metadata
def search_documents(query, top_k=3):
    # Perform entity recognition and topic modeling on the query
    entities = entity_recognition(query)
    query_embedding = retriever_tokenizer(query, return_tensors="pt", truncation=True, padding=True)
    query_embedding = retriever_model(**query_embedding).last_hidden_state.mean(dim=1).detach().numpy()
    
    distances, indices = faiss_index.search(query_embedding, top_k)
    
    retrieved_docs = [combined_documents[idx] for idx in indices[0]]
    return retrieved_docs

# Load pre-trained models and tokenizers
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

distilbert_model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")
distilbert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")

t5_model = T5ForConditionalGeneration.from_pretrained("t5-base")
t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")

gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2-large")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")

summarization_pipeline = pipeline("summarization", model=bart_model, tokenizer=bart_tokenizer)
qa_pipeline = pipeline("question-answering", model=distilbert_model, tokenizer=distilbert_tokenizer)
text_generation_pipeline = pipeline(
    "text-generation", 
    model=gpt2_model, 
    tokenizer=gpt2_tokenizer, 
    max_length=500,
    num_return_sequences=1, 
    temperature=0.7,
    top_p=0.9, 
    do_sample=True,
    truncation=True, 
    pad_token_id=gpt2_tokenizer.eos_token_id  
)

# Define the LangChain models using the Hugging Face pipelines
llm_summarization = HuggingFacePipeline(pipeline=summarization_pipeline)
llm_qa = HuggingFacePipeline(pipeline=qa_pipeline)
llm_text_generation = HuggingFacePipeline(pipeline=text_generation_pipeline)

# Define prompt templates
prompt_template_text_generation = PromptTemplate(
    input_variables=["context", "query"],
    template="{context}\n"
)

# Custom class for topic modeling
class TopicModelingChain:
    def run(self, inputs):
        docs = inputs["docs"]
        topics = topic_modeling(docs)
        return {"topics": topics}

# Custom class for entity recognition
class EntityRecognitionChain:
    def run(self, inputs):
        text = inputs["context"]
        entities = entity_recognition(text)
        return {"entities": entities}

# Create the chains for different tasks
topic_modeling_chain = TopicModelingChain()
entity_recognition_chain = EntityRecognitionChain()

response_chain_text_generation = LLMChain(
    llm=llm_text_generation,
    prompt=prompt_template_text_generation
)

summarization_chain = LLMChain(
    llm=llm_summarization,
    prompt=PromptTemplate(
        input_variables=["context"],
        template="{context}"
    )
)

qa_chain = LLMChain(
    llm=llm_qa,
    prompt=PromptTemplate(
        input_variables=["context", "query"],
        template="{context}\nQuestion: {query}\nAnswer:"
    )
)

# Combined chain with routing logic
class CombinedChain:
    def run(self, inputs):
        query = inputs["query"].lower()
        
        if "summarize" in query or "summary" in query:
            summary_result = summarization_chain.run({"context": inputs["context"]})
            inputs["context"] = summary_result
            response = None
        elif any(q_word in query for q_word in ["who", "what", "when", "where", "why", "how"]):
            qa_input = {"question": inputs["query"], "context": inputs["context"]}
            qa_result = qa_pipeline(qa_input)
            response = qa_result['answer']
        else:
            response = response_chain_text_generation.run(inputs)
        
        # Extract entities and topics from the context
        entities_result = entity_recognition_chain.run({"context": inputs["context"]})
        topics_result = topic_modeling_chain.run({"docs": [inputs["context"]]})
        
        # Include entities and topics in the final result
        result = {
            "summary": inputs.get("context", None) if "summarize" in query or "summary" in query else None,
            "response": response,
            "entities": entities_result.get("entities"),
            "topics": topics_result.get("topics")
        }
        return result

combined_chain = CombinedChain()

# Example search and response generation with validation
query = "What is the UK's policy for reducing emissions from heavy goods vehicles (HGVs) and aviation fuels?"
retrieved_docs = search_documents(query)
print("Retrieved Documents:", retrieved_docs)

combined_context = " ".join(retrieved_docs)
docs = retrieved_docs
inputs = {
    "context": combined_context,
    "query": query,
    "docs": docs
}

# Run the combined chain
result = combined_chain.run(inputs)
response = result["response"]
entities = result["entities"]
topics = result["topics"]

print("Generated Response:", response)
print("Extracted Entities:", entities)
print("Extracted Topics:", topics)
