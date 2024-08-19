import torch
from transformers import AutoTokenizer, AutoModel, T5Tokenizer, T5ForConditionalGeneration, pipeline
import faiss
import numpy as np
import spacy
import os
import json
import openai
from RAG.models.chunking_models import extract_entities, get_topic_distribution, summarize_documents
from RAG.models.llm_models import (
    retriever_model,
    retriever_tokenizer,
    summarizer_model,
    summarizer_tokenizer,
)

# Set your OpenAI API key here
openai.api_key = "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"

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

# Updated generate_response function to use GPT-3.5 via OpenAI API
def generate_response(query, countries=None):
    # Predefined responses for greetings and small talk
    greeting_responses = [
        "Hello! How can I assist you today?",
        "Hi there! What can I do for you?",
        "Greetings! How can I help you today?"
    ]
    
    thank_you_responses = [
        "You're welcome! Is there anything else you need?",
        "No problem! Happy to help.",
        "Glad I could assist. Anything else?"
    ]
    
    farewell_responses = [
        "Goodbye! Have a great day!",
        "Take care! If you need anything, feel free to ask.",
        "Bye! I'm here if you need more help."
    ]
    
    # Handling greetings
    if query.lower() in ["hello", "hi", "hey", "greetings"]:
        return greeting_responses[0]  # You can use random.choice(greeting_responses) for randomness
    
    # Handling thank you
    elif "thank" in query.lower():
        return thank_you_responses[0]  # Use random.choice(thank_you_responses) for variety
    
    # Handling farewells
    elif query.lower() in ["bye", "goodbye", "see you", "take care"]:
        return farewell_responses[0]  # You can also randomize this
    
    # For all other queries, proceed with the document retrieval and generation
    else:
        # Retrieve and summarize documents
        retrieved_docs = retrieve_documents(query, countries)
        summaries = summarize_documents(retrieved_docs)
        context = " ".join(summaries)

        # Ensure the context length does not exceed the model's max length
        max_input_length = 4096 - 500  # Assuming the max token length for GPT-3.5-turbo is 4096 tokens
        context = context[:max_input_length]

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
            )

            # Extract and return the response text
            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"An error occurred: {e}")
            return "Sorry, I couldn't generate a response."

# Example usage
query = "What tax incentives does Norway provide for electric vehicles?"
country = ["Norway"]
response = generate_response(query, country)
print("Response:")
print(response)
