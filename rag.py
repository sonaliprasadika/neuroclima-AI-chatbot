import torch
from transformers import AutoTokenizer, AutoModel, GPT2Tokenizer, GPT2LMHeadModel
import faiss
import numpy as np

# Load the retriever model and tokenizer
retriever_model_name = "bert-base-uncased"
retriever_tokenizer = AutoTokenizer.from_pretrained(retriever_model_name)
retriever_model = AutoModel.from_pretrained(retriever_model_name)

# Load the generator model and tokenizer
generator_model_name = "gpt2"
generator_tokenizer = GPT2Tokenizer.from_pretrained(generator_model_name)
generator_model = GPT2LMHeadModel.from_pretrained(generator_model_name)

# Example documents
documents = [
    "The capital of France is Paris.",
    "The moon orbits the Earth.",
    "The largest ocean on Earth is the Pacific Ocean."
]

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
    distances, indices = index.search(query_embedding, top_k)
    return [documents[i] for i in indices[0]]

def generate_response(query):
    retrieved_docs = retrieve_documents(query)
    augmented_query = query + " " + " ".join(retrieved_docs)
    inputs = generator_tokenizer.encode(augmented_query, return_tensors="pt")
    outputs = generator_model.generate(inputs, max_length=150)
    response = generator_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Example usage
query = "What is the largest ocean?"
response = generate_response(query)
print(response)
