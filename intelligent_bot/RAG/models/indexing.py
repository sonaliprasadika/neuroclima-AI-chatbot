import torch
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np
import os
import json
import pandas as pd

# Assuming the functions extract_entities, get_topic_distribution, summarize_documents are defined
from chunking_models import extract_entities, get_topic_distribution, summarize_documents
from llm_models import (
    retriever_model,
    retriever_tokenizer,
    summarizer_model,
    summarizer_tokenizer,
)

# Load the dataset
df = pd.read_csv('../../dataset/combined_dataset.csv', encoding='latin1')
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
    # Assuming extract_entities and get_topic_distribution are defined and return the correct format
    document_entities = extract_entities(combined_documents[idx])
    document_topics = get_topic_distribution(combined_documents[idx])
    
    # Convert NumPy float32 to Python float for JSON serialization
    topics_converted = {str(topic): float(weight) for topic, weight in document_topics.items()}
    
    metadata[str(idx)] = {
        'document': combined_documents[idx],
        'entities': document_entities,
        'topics': topics_converted
    }

# Save metadata as JSON
with open('saved_data/metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, ensure_ascii=False, indent=4)


# from elasticsearch import Elasticsearch, helpers
# import json

# # Initialize Elasticsearch client
# es = Elasticsearch([{'host': 'http://195.148.31.180:9200', 'port': 9200}])

# # Define the index name
# index_name = 'indexed_policy_descriptions'

# # Define the index settings and mappings
# index_settings = {
#     "settings": {
#         "number_of_shards": 1,
#         "number_of_replicas": 1
#     },
#     "mappings": {
#         "properties": {
#             "document": {"type": "text"},
#             "entities": {"type": "keyword"},
#             "topics": {"type": "object"},
#             "embedding": {
#                 "type": "dense_vector",
#                 "dims": document_embeddings.shape[1]  # embedding dimensions
#             }
#         }
#     }
# }

# # Create the index if it doesn't exist
# if not es.indices.exists(index=index_name):
#     es.indices.create(index=index_name, body=index_settings)

# # Prepare documents for bulk ingestion
# actions = []
# for idx in document_ids:
#     action = {
#         "_index": index_name,
#         "_id": str(idx),
#         "_source": {
#             "document": combined_documents[idx],
#             "entities": document_entities[idx],
#             "topics": metadata[str(idx)]['topics'],
#             "embedding": document_embeddings[idx].tolist()  # Convert to list for JSON serialization
#         }
#     }
#     actions.append(action)

# # Bulk ingest data into Elasticsearch
# helpers.bulk(es, actions)

# print(f"Data indexed into Elasticsearch index '{index_name}'.")
