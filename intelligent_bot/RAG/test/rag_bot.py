import string
import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from transformers import (
    BartForConditionalGeneration, BartTokenizer,
    DistilBertForQuestionAnswering, DistilBertTokenizer,
    T5ForConditionalGeneration, T5Tokenizer,
    GPT2LMHeadModel, GPT2Tokenizer,
    pipeline
)
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate

import spacy
from gensim import corpora
from gensim.models import LdaModel
import logging

# Configure logging
logging.basicConfig(filename='bulk_index_errors.log', level=logging.ERROR)

# Load the dataset
df = pd.read_csv('dataset/combined_dataset.csv', encoding='latin1')
print(df.head())

# Connect to Elasticsearch
es = Elasticsearch("http://195.148.31.180:9200")

# Define the index
index_name = "policy_descriptions"

# Index documents
def index_documents(df):
    actions = [
        {
            "_index": index_name,
            "_source": {
                "country": row["country"],
                "policy_description": row["policy_description"]
            }
        }
        for idx, row in df.iterrows()
    ]
    
    success, failed = bulk(es, actions, raise_on_error=False)
    
    if failed:
        logging.error(f"{len(failed)} documents failed to index.")
        for fail in failed:
            logging.error(f"Error: {fail}")
            # Handle specific error cases
            if 'error' in fail:
                error = fail['error']
                if 'type' in error and error['type'] == 'mapper_parsing_exception':
                    # Handle mapping errors
                    logging.error(f"Mapping error: {error['reason']}")
                elif 'type' in error and error['type'] == 'illegal_argument_exception':
                    # Handle illegal argument errors
                    logging.error(f"Illegal argument: {error['reason']}")
                # Add other error handling as needed

        # Retry indexing failed documents
        retry_actions = [fail["create"] for fail in failed if "create" in fail]
        if retry_actions:
            bulk(es, retry_actions)

# index_documents(df)

# Search documents
def search_documents(query, index_name="policy_descriptions", size=3):
    search_query = {
        "query": {
            "match": {
                "policy_description": query
            }
        },
        "size": size
    }
    response = es.search(index=index_name, body=search_query)
    return [hit["_source"] for hit in response["hits"]["hits"]]

# Load pre-trained models and tokenizers
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

distilbert_model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")
distilbert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")

t5_model = T5ForConditionalGeneration.from_pretrained("t5-base")
t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")

gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2-large")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")

summarization_pipeline = pipeline("summarization", model=bart_model, tokenizer=bart_tokenizer)
qa_pipeline = pipeline("question-answering", model=distilbert_model, tokenizer=distilbert_tokenizer)
text_generation_pipeline = pipeline(
    "text-generation", 
    model=gpt2_model, 
    tokenizer=gpt2_tokenizer, 
    max_length=500,  # Reduced max_length to avoid overly verbose outputs
    num_return_sequences=1, 
    temperature=0.7,  # Lower temperature for more focused and deterministic output
    top_p=0.9, 
    do_sample=True,
    truncation=True, 
    pad_token_id=gpt2_tokenizer.eos_token_id  
)

# Define the LangChain models using the Hugging Face pipelines
llm_summarization = HuggingFacePipeline(pipeline=summarization_pipeline)
llm_qa = HuggingFacePipeline(pipeline=qa_pipeline)
llm_text_generation = HuggingFacePipeline(pipeline=text_generation_pipeline)

# Define prompt templates
prompt_template_text_generation = PromptTemplate(
    input_variables=["context", "query"],
    template="{context}\n"
)

# Initialize spaCy model for entity recognition
nlp = spacy.load("en_core_web_sm")

# Define a function for entity recognition
def entity_recognition(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

# Define a function for topic modeling with enhanced preprocessing
def topic_modeling(docs, num_topics=3, num_words=4):
    stopwords = set(nlp.Defaults.stop_words)
    texts = []
    for doc in docs:
        tokens = doc.lower().split()
        tokens = [word.strip(string.punctuation) for word in tokens if word not in stopwords and len(word) > 1]
        texts.append(tokens)
        
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
    
    topics = []
    for i, topic in lda.show_topics(formatted=False, num_words=num_words):
        topics.append([word for word, _ in topic if word not in stopwords and len(word) > 1])
    
    return topics

# Custom class for topic modeling
class TopicModelingChain:
    def run(self, inputs):
        docs = inputs["docs"]
        topics = topic_modeling(docs)
        return {"topics": topics}

# Custom class for entity recognition
class EntityRecognitionChain:
    def run(self, inputs):
        text = inputs["context"]
        entities = entity_recognition(text)
        return {"entities": entities}

# Create the chains for different tasks
topic_modeling_chain = TopicModelingChain()
entity_recognition_chain = EntityRecognitionChain()

response_chain_text_generation = LLMChain(
    llm=llm_text_generation,
    prompt=prompt_template_text_generation
)

summarization_chain = LLMChain(
    llm=llm_summarization,
    prompt=PromptTemplate(
        input_variables=["context"],
        template="{context}"
    )
)

qa_chain = LLMChain(
    llm=llm_qa,
    prompt=PromptTemplate(
        input_variables=["context", "query"],
        template="{context}\nQuestion: {query}\nAnswer:"
    )
)

# Combined chain with routing logic
class CombinedChain:
    def run(self, inputs):
        query = inputs["query"].lower()
        
        if "summarize" in query or "summary" in query:
            summary_result = summarization_chain.run({"context": inputs["context"]})
            inputs["context"] = summary_result
            response = None
        elif any(q_word in query for q_word in ["who", "what", "when", "where", "why", "how"]):
            qa_input = {"question": inputs["query"], "context": inputs["context"]}
            qa_result = qa_pipeline(qa_input)
            response = qa_result['answer']
        else:
            response = response_chain_text_generation.run(inputs)
        
        result = {
            "summary": inputs.get("context", None) if "summarize" in query or "summary" in query else None,
            "response": response
        }
        return result

combined_chain = CombinedChain()

# Example search and response generation
query = "Explain the impact of renewable energy on global warming."
retrieved_docs = search_documents(query)
print("Retrieved Documents:", retrieved_docs)

# Combine retrieved documents
combined_context = " ".join([doc["policy_description"] for doc in retrieved_docs])
docs = [doc["policy_description"] for doc in retrieved_docs]
inputs = {
    "context": combined_context,
    "query": query,
    "docs": docs
}

# Run the combined chain
result = combined_chain.run(inputs)
print("Generated Response:", result["response"])

# # Combined chain with routing logic
# class CombinedChain:
#     def run(self, inputs):
#         query = inputs["query"].lower()
        
#         if "summarize" in query or "summary" in query:
#             summary_result = summarization_chain.run({"context": inputs["context"]})
#             inputs["context"] = summary_result
#             response = None
#         elif any(q_word in query for q_word in ["who", "what", "when", "where", "why", "how"]):
#             qa_input = {"question": inputs["query"], "context": inputs["context"]}
#             qa_result = qa_pipeline(qa_input)
#             response = qa_result['answer']
#         else:
#             response = response_chain_text_generation.run(inputs)
        
#         # topics = topic_modeling_chain.run({"docs": inputs["docs"]})
#         # entities = entity_recognition_chain.run({"context": inputs["context"]})
        
#         result = {
#             # "topics": topics["topics"],
#             # "entities": entities["entities"],
#             "summary": inputs.get("context", None) if "summarize" in query or "summary" in query else None,
#             "response": response
#         }
#         return result

# combined_chain = CombinedChain()

# # Example search and response generation
# query = "Explain the impact of renewable energy on global warming."
# retrieved_docs = search_documents(query)
# print("Retrieved Documents:", retrieved_docs)

# # Combine retrieved documents
# combined_context = " ".join([doc["policy_description"] for doc in retrieved_docs])
# docs = [doc["policy_description"] for doc in retrieved_docs]
# inputs = {
#     "context": combined_context,
#     "query": query,
#     "docs": docs
# }

# # Run the combined chain
# result = combined_chain.run(inputs)
# # print("Topics:", result["topics"])
# # print("Entities:", result["entities"])
# # print("Summary:", result["summary"])
# print("Generated Response:", result["response"])

