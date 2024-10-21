import torch
from transformers import AutoTokenizer, AutoModel

# Load Retriever Model and Tokenizer
retriever_model_name = "bert-base-uncased"
retriever_tokenizer = AutoTokenizer.from_pretrained(retriever_model_name)
retriever_model = AutoModel.from_pretrained(retriever_model_name)
