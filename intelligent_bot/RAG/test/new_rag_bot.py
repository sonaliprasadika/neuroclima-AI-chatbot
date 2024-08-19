import torch
from transformers import AutoTokenizer, AutoModel, GPT2Tokenizer, GPT2LMHeadModel, T5Tokenizer, T5ForConditionalGeneration, pipeline
import faiss
import numpy as np
import spacy
import os
import json
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from models.chunking_models import extract_entities, get_topic_distribution, summarize_documents
from models.llm_models import (
    retriever_model,
    retriever_tokenizer,
    generator_model,
    generator_tokenizer,
    summarizer_model,
    summarizer_tokenizer,
)
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
    print("I am langchain program")
    return [metadata[str(i)]['document'] for _, i in doc_scores[:top_k]]

# Modify the generate_response function to use LangChain
def generate_response(query, countries=None):
    # Retrieve and summarize documents
    retrieved_docs = retrieve_documents(query, countries)
    summaries = summarize_documents(retrieved_docs)
    context = " ".join(summaries)
    
    # Ensure the context length does not exceed the model's max length
    max_input_length = generator_tokenizer.model_max_length - 20
    context = generator_tokenizer.decode(generator_tokenizer.encode(context, max_length=max_input_length, truncation=True))

    # Create a Hugging Face pipeline for the generator model
    generator_pipeline = pipeline('text-generation', model=generator_model, tokenizer=generator_tokenizer, max_new_tokens=150)

    # Wrap the Hugging Face pipeline in a LangChain-compatible interface
    langchain_pipeline = HuggingFacePipeline(pipeline=generator_pipeline)

    # Define a prompt template
    prompt_template = PromptTemplate(
        input_variables=["query", "context"], 
        template="Question: {query}\n\nContext: {context}\n\nAnswer:"
    )

    # Initialize the LLMChain with the prompt template and generator pipeline
    llm_chain = LLMChain(prompt=prompt_template, llm=langchain_pipeline)

    # Generate the response using LangChain
    response = llm_chain.run({"query": query, "context": context})
    
    return response

# Example usage
query = "What tax incentives does Norway provide for electric vehicles?"
country = ["Norway"]
response = generate_response(query, country)
print("Response:")
print(response)