import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load Summarizer Model and Tokenizer
summarizer_model_name = "t5-small"
summarizer_tokenizer = T5Tokenizer.from_pretrained(summarizer_model_name)
summarizer_model = T5ForConditionalGeneration.from_pretrained(summarizer_model_name)
