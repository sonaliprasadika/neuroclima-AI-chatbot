import os
import json  
import numpy as np
import pandas as pd
import spacy
import gensim
from gensim import corpora
import faiss
import wikipediaapi
from models.llm_models import (
    summarizer_model,
    summarizer_tokenizer,
)

# Load the dataset
df = pd.read_csv('../dataset/combined_dataset_training.csv', encoding='latin1')
print(df.head())

# Extract countries and policy descriptions
countries = df['country'].tolist()
documents = df['policy_description'].tolist()

# Combine countries and policy descriptions
combined_documents = [f"{country}: {policy}" for country, policy in zip(countries, documents)]

# Preprocessing
def preprocess_text(texts):
    return [[word for word in doc.lower().split() if word.isalnum()] for doc in texts]

preprocessed_docs = preprocess_text(combined_documents)
dictionary = corpora.Dictionary(preprocessed_docs)
corpus = [dictionary.doc2bow(doc) for doc in preprocessed_docs]

# Entity Recognition
nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

# Topic Modeling
lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=2, id2word=dictionary, passes=15)

def get_topic_distribution(text):
    bow = dictionary.doc2bow(preprocess_text([text])[0])
    return dict(lda_model.get_document_topics(bow))

def summarize_documents(docs, max_length=150):
    summaries = []
    for doc in docs:
        input_text = "summarize: " + doc
        inputs = summarizer_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = summarizer_model.generate(inputs, max_length=max_length, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
    return summaries
