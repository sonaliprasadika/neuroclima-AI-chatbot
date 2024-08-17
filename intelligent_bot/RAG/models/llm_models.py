import torch
from transformers import AutoTokenizer, AutoModel, T5Tokenizer, T5ForConditionalGeneration

# Load Retriever Model and Tokenizer
retriever_model_name = "bert-base-uncased"
retriever_tokenizer = AutoTokenizer.from_pretrained(retriever_model_name)
retriever_model = AutoModel.from_pretrained(retriever_model_name)

# # Load Generator Model and Tokenizer
# # generator_model_name = "gpt2"
# generator_model_name = "gpt-3.5-turbo"
# generator_tokenizer = GPT2Tokenizer.from_pretrained(generator_model_name)
# generator_model = GPT2LMHeadModel.from_pretrained(generator_model_name)

# Load Summarizer Model and Tokenizer
summarizer_model_name = "t5-small"
summarizer_tokenizer = T5Tokenizer.from_pretrained(summarizer_model_name)
summarizer_model = T5ForConditionalGeneration.from_pretrained(summarizer_model_name)
