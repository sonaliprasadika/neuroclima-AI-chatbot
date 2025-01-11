import torch
from transformers import AutoTokenizer, AutoModel, T5Tokenizer, T5ForConditionalGeneration, pipeline
import faiss
import numpy as np
import spacy
import os
import json
from models.chunking_models import extract_entities, get_topic_distribution, summarize_documents
from models.llm_models import (
    retriever_model,
    retriever_tokenizer,
    # generator_model,
    # generator_tokenizer,
    summarizer_model,
    summarizer_tokenizer,
)
import openai

# Set your OpenAI API key here
openai.api_key = ""

# Load the FAISS index and metadata from files
index_with_ids = faiss.read_index('saved_data/retriever_index_with_ids.faiss')

with open('saved_data/metadata.json', 'r', encoding='utf-8') as f:
    metadata = json.load(f)

# Function to retrieve documents based on the query
def retrieve_documents(query, countries=None, top_k=4):
    # Tokenize and encode the query using the retriever model
    inputs = retriever_tokenizer(query, return_tensors="pt", truncation=True, padding=True)
    outputs = retriever_model(**inputs)
    query_embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()

    # Search the FAISS index using the query embedding
    distances, indices = index_with_ids.search(query_embedding, len(metadata))

    query_topics = get_topic_distribution(query)
    query_entities = extract_entities(query)

    # Score and rank the retrieved documents
    doc_scores = []
    for i in range(len(indices[0])):
        idx = indices[0][i]
        doc_metadata = metadata[str(idx)]

        if countries and not any(country.lower() in doc_metadata['document'].lower() for country in countries):
            continue

        doc_topic_score = sum(query_topics.get(str(topic), 0) * weight for topic, weight in doc_metadata['topics'].items())
        doc_entity_score = len(set(ent[0] for ent in query_entities) & set(ent[0] for ent in doc_metadata['entities']))

        total_score = 0.5 * distances[0][i] - 2.0 * doc_topic_score + 2.0 * doc_entity_score

        if countries and any(country.lower() in doc_metadata['document'].lower() for country in countries):
            total_score *= 0.5

        doc_scores.append((total_score, idx))

    # Sort and return the top-k documents
    doc_scores.sort(key=lambda x: x[0])
    return [metadata[str(i)]['document'] for _, i in doc_scores[:top_k]]

def generate_response(query, countries=None):
    # Retrieve and summarize documents based on the query
    retrieved_docs = retrieve_documents(query, countries)
    summaries = summarize_documents(retrieved_docs)
    
    # Concatenate summaries into a single context
    context = " ".join(summaries)
    
    # Ensure the context does not exceed the model's max length
    max_input_length = 4096 - 50  # Adjust based on the total token limit
    context = context[:max_input_length]  # Truncate if necessary

    # Augment the query with the context
    if countries:
        country_list = ", ".join(countries)
        augmented_query = f"Question: {query}\n\nRelevant information extracted from documents related to {country_list}:\n\n{context}\n\nAnswer:"
    else:
        augmented_query = f"Question: {query}\n\nRelevant information extracted from documents:\n\n{context}\n\nAnswer:"
    
    # Generate a response using the OpenAI API
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": augmented_query}
            ],
            max_tokens=500,  # Adjust the number of tokens as needed
            temperature=0.7,
            top_p=0.95,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        return context, response['choices'][0]['message']['content']
    except Exception as e:
        print(f"An error occurred: {e}")
        return context, "Sorry, I couldn't generate a response."

# Example usage
query = "What measures has Norway implemented to reduce methane emissions?"
country = ["Norway"]
# country = ["United Kingdom"]
context, response  = generate_response(query, country)
print("Context:", context)
print("Response:", response)
