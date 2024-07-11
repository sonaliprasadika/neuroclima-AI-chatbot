# from elasticsearch import Elasticsearch
# from elasticsearch.helpers import bulk
# from transformers import (
#     BartForConditionalGeneration, BartTokenizer,
#     DistilBertForQuestionAnswering, DistilBertTokenizer,
#     T5ForConditionalGeneration, T5Tokenizer,
#     pipeline
# )
# from langchain.chains import LLMChain, SimpleSequentialChain
# from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
# from langchain.prompts import PromptTemplate

# import spacy
# from gensim import corpora
# from gensim.models import LdaModel

# # Connect to Elasticsearch
# es = Elasticsearch("http://195.148.31.180:9200")

# # Example documents to index
# documents = [
#     {"title": "Document 1", "content": "This is the content of document 1."},
#     {"title": "Document 2", "content": "This is the content of document 2."},
#     {"title": "Document 3", "content": "This is the content of document 3."}
# ]

# # Define the index
# index_name = "weather_responses"

# # Index documents
# def index_documents(documents):
#     actions = [
#         {
#             "_index": index_name,
#             "_source": doc
#         }
#         for doc in documents
#     ]
#     bulk(es, actions)

# index_documents(documents)

# # Search documents
# def search_documents(query, index_name="weather_responses", size=3):
#     search_query = {
#         "query": {
#             "match": {
#                 "content": query
#             }
#         },
#         "size": size
#     }
#     response = es.search(index=index_name, body=search_query)
#     return [hit["_source"] for hit in response["hits"]["hits"]]

# # Load pre-trained models and tokenizers
# bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
# bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

# distilbert_model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")
# distilbert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")

# t5_model = T5ForConditionalGeneration.from_pretrained("t5-base")
# t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")

# # Define pipelines
# summarization_pipeline = pipeline("summarization", model=bart_model, tokenizer=bart_tokenizer)
# qa_pipeline = pipeline("question-answering", model=distilbert_model, tokenizer=distilbert_tokenizer)
# text_generation_pipeline = pipeline("text2text-generation", model=t5_model, tokenizer=t5_tokenizer, max_new_tokens=150)

# # Define the LangChain models using the Hugging Face pipelines
# llm_summarization = HuggingFacePipeline(pipeline=summarization_pipeline)
# llm_qa = HuggingFacePipeline(pipeline=qa_pipeline)
# llm_text_generation = HuggingFacePipeline(pipeline=text_generation_pipeline)

# # Define prompt templates
# prompt_template_text_generation = PromptTemplate(
#     input_variables=["context", "query"],
#     template="Context: {context}\n\nQ: {query}\nA:"
# )

# # Initialize spaCy model for entity recognition
# nlp = spacy.load("en_core_web_sm")

# # Define a function for entity recognition
# def entity_recognition(text):
#     doc = nlp(text)
#     return [(ent.text, ent.label_) for ent in doc.ents]

# # Define a function for topic modeling
# def topic_modeling(docs):
#     texts = [doc.split() for doc in docs]
#     dictionary = corpora.Dictionary(texts)
#     corpus = [dictionary.doc2bow(text) for text in texts]
#     lda = LdaModel(corpus, num_topics=1, id2word=dictionary, passes=15)
#     topics = lda.print_topics(num_words=4)
#     return topics

# # Custom class for topic modeling
# class TopicModelingChain:
#     def run(self, inputs):
#         docs = inputs["docs"]
#         topics = topic_modeling(docs)
#         return {"topics": topics}

# # Custom class for entity recognition
# class EntityRecognitionChain:
#     def run(self, inputs):
#         text = inputs["context"]
#         entities = entity_recognition(text)
#         return {"entities": entities}

# # Create the chains for different tasks
# topic_modeling_chain = TopicModelingChain()
# entity_recognition_chain = EntityRecognitionChain()

# response_chain_text_generation = LLMChain(
#     llm=llm_text_generation,
#     prompt=prompt_template_text_generation
# )

# summarization_chain = LLMChain(
#     llm=llm_summarization,
#     prompt=PromptTemplate(
#         input_variables=["context"],
#         template="{context}"
#     )
# )

# qa_chain = LLMChain(
#     llm=llm_qa,
#     prompt=PromptTemplate(
#         input_variables=["context", "query"],
#         template="{context}\nQuestion: {query}\nAnswer:"
#     )
# )

# # Combined chain with routing logic
# class CombinedChain:
#     def run(self, inputs):
#         query = inputs["query"].lower()
        
#         if "summarize" in query or "summary" in query:
#             # Use summarization chain
#             summary_result = summarization_chain.run({"context": inputs["context"]})
#             inputs["context"] = summary_result
#             response = None  # No need for response generation
#         elif any(q_word in query for q_word in ["who", "what", "when", "where", "why", "how"]):
#             # Use QA chain
#             qa_result = qa_chain.run({"context": inputs["context"], "query": inputs["query"]})
#             response = qa_result
#         else:
#             # Use text generation chain
#             response = response_chain_text_generation.run(inputs)
        
#         # Perform topic modeling
#         topics = topic_modeling_chain.run({"docs": inputs["docs"]})
#         # Perform entity recognition
#         entities = entity_recognition_chain.run({"context": inputs["context"]})
        
#         # Combine results
#         result = {
#             "topics": topics["topics"],
#             "entities": entities["entities"],
#             "summary": inputs.get("context", None) if "summarize" in query or "summary" in query else None,
#             "response": response
#         }
#         return result

# combined_chain = CombinedChain()

# # Example search and response generation
# query = "content of document 1"
# retrieved_docs = search_documents(query)
# print("Retrieved Documents:", retrieved_docs)

# # Combine retrieved documents
# combined_context = " ".join([doc["content"] for doc in retrieved_docs])
# docs = [doc["content"] for doc in retrieved_docs]
# inputs = {
#     "context": combined_context,
#     "query": query,
#     "docs": docs
# }

# # Run the combined chain
# result = combined_chain.run(inputs)
# print("Topics:", result["topics"])
# print("Entities:", result["entities"])
# print("Summary:", result["summary"])
# print("Generated Response:", result["response"])

import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from transformers import (
    BartForConditionalGeneration, BartTokenizer,
    DistilBertForQuestionAnswering, DistilBertTokenizer,
    T5ForConditionalGeneration, T5Tokenizer,
    pipeline
)
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate

import spacy
from gensim import corpora
from gensim.models import LdaModel

# Load the dataset
df = pd.read_csv('combined_dataset_training.csv', encoding='latin1')
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
    bulk(es, actions)

index_documents(df)

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

# Define pipelines
summarization_pipeline = pipeline("summarization", model=bart_model, tokenizer=bart_tokenizer)
qa_pipeline = pipeline("question-answering", model=distilbert_model, tokenizer=distilbert_tokenizer)
text_generation_pipeline = pipeline("text2text-generation", model=t5_model, tokenizer=t5_tokenizer, max_new_tokens=150)

# Define the LangChain models using the Hugging Face pipelines
llm_summarization = HuggingFacePipeline(pipeline=summarization_pipeline)
llm_qa = HuggingFacePipeline(pipeline=qa_pipeline)
llm_text_generation = HuggingFacePipeline(pipeline=text_generation_pipeline)

# Define prompt templates
prompt_template_text_generation = PromptTemplate(
    input_variables=["context", "query"],
    template="Context: {context}\n\nQ: {query}\nA:"
)

# Initialize spaCy model for entity recognition
nlp = spacy.load("en_core_web_sm")

# Define a function for entity recognition
def entity_recognition(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

# Define a function for topic modeling
def topic_modeling(docs):
    texts = [doc.split() for doc in docs]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda = LdaModel(corpus, num_topics=1, id2word=dictionary, passes=15)
    topics = lda.print_topics(num_words=4)
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
            # Use summarization chain
            summary_result = summarization_chain.run({"context": inputs["context"]})
            inputs["context"] = summary_result
            response = None  # No need for response generation
        elif any(q_word in query for q_word in ["who", "what", "when", "where", "why", "how"]):
            # Use QA pipeline directly with the correct format
            qa_input = {"question": inputs["query"], "context": inputs["context"]}
            qa_result = qa_pipeline(qa_input)
            response = qa_result['answer']
        else:
            # Use text generation chain
            response = response_chain_text_generation.run(inputs)
        
        # Perform topic modeling
        topics = topic_modeling_chain.run({"docs": inputs["docs"]})
        # Perform entity recognition
        entities = entity_recognition_chain.run({"context": inputs["context"]})
        
        # Combine results
        result = {
            "topics": topics["topics"],
            "entities": entities["entities"],
            "summary": inputs.get("context", None) if "summarize" in query or "summary" in query else None,
            "response": response
        }
        return result

combined_chain = CombinedChain()

# Example search and response generation
query = "What are the latest policies related to carbon emissions?"
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
print("Topics:", result["topics"])
print("Entities:", result["entities"])
print("Summary:", result["summary"])
print("Generated Response:", result["response"])

