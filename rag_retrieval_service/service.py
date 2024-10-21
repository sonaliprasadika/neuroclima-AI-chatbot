import torch
import faiss
import numpy as np
import spacy
import os
import json
import random
import time
import re
from elasticsearch import Elasticsearch, exceptions
from elasticsearch.exceptions import TransportError  
from elasticsearch.helpers import bulk
from models.chunking_models import extract_entities, get_topic_distribution
from models.llm_models import (
    retriever_model,
    retriever_tokenizer,
)

es = Elasticsearch("http://195.148.31.180:9200") 

# Load the FAISS index
index_with_ids = faiss.read_index('saved_data/retriever_index_with_ids.faiss')

def retrieve_documents(query, countries=None, top_k=4):
    # Tokenize and encode the query using the retriever model
    inputs = retriever_tokenizer(query, return_tensors="pt", truncation=True, padding=True)
    outputs = retriever_model(**inputs)
    query_embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()

    # Search the FAISS index using the query embedding
    distances, indices = index_with_ids.search(query_embedding, top_k)

    query_topics = get_topic_distribution(query)
    query_entities = extract_entities(query)

    # Initialize document scores list
    doc_scores = []

    # Fetch data from Elasticsearch
    for i in range(len(indices[0])):
        idx = indices[0][i]
        # Query Elasticsearch by ID (assuming IDs are used similarly to metadata keys)
        res = es.get(index="test123", id=idx)
        doc_metadata = res["_source"]
        
        # Filter by countries if applicable
        if countries and not any(country.lower() in doc_metadata['document'].lower() for country in countries):
            continue

        # Calculate topic and entity scores
        doc_topic_score = sum(query_topics.get(str(topic), 0) * float(doc_metadata['topics'][str(topic)]) for topic in doc_metadata['topics'])
        doc_entity_score = len(set(ent[0] for ent in query_entities) & set(doc_metadata['entities']))

        # Calculate total score
        total_score = 0.5 * distances[0][i] - 2.0 * doc_topic_score + 2.0 * doc_entity_score

        # Apply additional scoring logic if countries are matched
        if countries and any(country.lower() in doc_metadata['document'].lower() for country in countries):
            total_score *= 0.5

        doc_scores.append((total_score, idx))

    # Sort and return the top-k documents based on their scores
    doc_scores.sort(key=lambda x: x[0])
    return [es.get(index="test123", id=str(i))["_source"]["document"] for _, i in doc_scores[:top_k]]