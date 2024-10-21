import os
import json  
import numpy as np
import pandas as pd
import gensim
from gensim import corpora
import wikipediaapi
from models.llm_models import (
    summarizer_model,
    summarizer_tokenizer,
)

def summarize_documents(docs, max_length=150):
    summaries = []
    for doc in docs:
        input_text = "summarize: " + doc
        inputs = summarizer_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = summarizer_model.generate(inputs, max_length=max_length, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
    return summaries
