import torch
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np
import os
import json
import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

# Assuming the functions extract_entities, get_topic_distribution, summarize_documents are defined
from chunking_models import extract_entities, get_topic_distribution, summarize_documents
from llm_models import (
    retriever_model,
    retriever_tokenizer,
    summarizer_model,
    summarizer_tokenizer,
)

# Load the dataset
df = pd.read_csv('../../dataset/combined_dataset_training.csv', encoding='latin1')
print(df.head())

# Extract countries and policy descriptions
countries = df['country'].tolist()
documents = df['policy_description'].tolist()

# Combine countries and policy descriptions
combined_documents = [f"{country}: {policy}" for country, policy in zip(countries, documents)]

# Encode documents using the retriever model
document_embeddings = []
for doc in combined_documents:
    inputs = retriever_tokenizer(doc, return_tensors="pt", truncation=True, padding=True)
    # Move inputs to GPU if available
    if torch.cuda.is_available():
        inputs = {key: value.cuda() for key, value in inputs.items()}
        retriever_model.cuda()
    outputs = retriever_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    document_embeddings.append(embeddings.detach().cpu().numpy())  # Move to CPU before converting to NumPy

document_embeddings = np.vstack(document_embeddings)

# Assign unique IDs to each document
document_ids = np.arange(len(combined_documents))

# Debugging Step: Check the lengths of various data structures
print(f"Number of combined documents: {len(combined_documents)}")
print(f"Number of document embeddings: {len(document_embeddings)}")

# Create a FAISS index with ID mapping
dimension = document_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index_with_ids = faiss.IndexIDMap(index)
index_with_ids.add_with_ids(document_embeddings, document_ids)

# Create directory if it does not exist
os.makedirs('saved_data', exist_ok=True)

# Save the FAISS index with IDs to a file in the 'saved_data' directory
faiss.write_index(index_with_ids, 'saved_data/retriever_index_with_ids.faiss')

# Prepare metadata mapping and convert keys to strings
metadata = {}
for idx in document_ids:
    try:
        # Extract entities and topics for each document
        document_entities = extract_entities(combined_documents[idx])
        document_topics = get_topic_distribution(combined_documents[idx])

        # Debugging Step: Check for potential issues
        if document_entities is None or document_topics is None:
            print(f"Missing data for document ID {idx}. Skipping.")
            continue

        # Convert NumPy float32 to Python float for JSON serialization
        topics_converted = {str(topic): float(weight) for topic, weight in document_topics.items()}

        # Store metadata
        metadata[str(idx)] = {
            'document': combined_documents[idx],
            'entities': document_entities,
            'topics': topics_converted
        }
    except IndexError as e:
        print(f"Error processing document ID {idx}: {e}")
        continue

# Save metadata as JSON
with open('saved_data/metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, ensure_ascii=False, indent=4)

# Connect to Elasticsearch
es = Elasticsearch("http://195.148.31.180:9200")  # Replace with your actual Elasticsearch endpoint

# Define the index name
INDEX_NAME = "test123"

# Create the index if it doesn't exist
if not es.indices.exists(index=INDEX_NAME):
    es.indices.create(index=INDEX_NAME, body={
        "mappings": {
            "properties": {
                "document": {"type": "text"},
                "entities": {"type": "nested"},  # Store entities as a nested field
                "topics": {"type": "nested"},    # Store topics as a nested field
                "embedding": {"type": "dense_vector", "dims": 768}  # Store document embeddings
            }
        }
    })

# Ensure that only valid data is indexed
actions = []
for idx, doc in metadata.items():
    try:
        action = {
            "_index": INDEX_NAME,
            "_id": idx,
            "_source": {
                "document": doc['document'],
                "entities": doc['entities'],
                "topics": doc['topics'],
                "embedding": document_embeddings[int(idx)].tolist()  # Convert to list for JSON serialization
            }
        }
        actions.append(action)
    except KeyError as e:
        print(f"Error with data for document ID {idx}: {e}")
        continue

# Bulk ingest data into Elasticsearch
bulk(es, actions)

print(f"Data indexed into Elasticsearch index '{INDEX_NAME}'.")