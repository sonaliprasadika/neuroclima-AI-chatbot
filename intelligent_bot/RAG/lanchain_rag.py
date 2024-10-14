# import torch
# from transformers import AutoTokenizer, AutoModel, T5Tokenizer, T5ForConditionalGeneration, pipeline
# import faiss
# import numpy as np
# import spacy
# import os
# import json
# import openai
# import random
# from dotenv import load_dotenv
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
# from RAG.models.chunking_models import extract_entities, get_topic_distribution, summarize_documents
# from RAG.models.llm_models import (
#     retriever_model,
#     retriever_tokenizer,
#     summarizer_model,
#     summarizer_tokenizer,
# )
# from langchain.llms import OpenAI
# import time
# import re
# # Load the FAISS index and metadata from files
# index_with_ids = faiss.read_index('saved_data/retriever_index_with_ids.faiss')

# with open('saved_data/metadata.json', 'r', encoding='utf-8') as f:
#     metadata = json.load(f)

# # Load environment variables from .env file
# load_dotenv()

# # Set the API key using the environment variable
# openai.api_key = os.getenv("OPENAI_API_KEY")

# # Define a custom LLM class for OpenAI GPT-3.5
# def OpenAI_GPT3_5_LLM(model, max_tokens=100, temperature=0.7, top_p=1):
#     # Return a LangChain OpenAI LLM instance configured with the provided parameters
#     return OpenAI(
#         model_name=model,
#         max_tokens=max_tokens,
#         temperature=temperature,
#         top_p=top_p
#     )
    
# # Function to retrieve documents based on the query
# def retrieve_documents(query, countries=None, top_k=4):
#     # Tokenize and encode the query using the retriever model
#     inputs = retriever_tokenizer(query, return_tensors="pt", truncation=True, padding=True)
#     outputs = retriever_model(**inputs)
#     query_embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()

#     # Search the FAISS index using the query embedding
#     distances, indices = index_with_ids.search(query_embedding, len(metadata))

#     query_topics = get_topic_distribution(query)
#     query_entities = extract_entities(query)

#     # Score and rank the retrieved documents
#     doc_scores = []
#     for i in range(len(indices[0])):
#         idx = indices[0][i]
#         doc_metadata = metadata[str(idx)]

#         if countries and not any(country.lower() in doc_metadata['document'].lower() for country in countries):
#             continue

#         doc_topic_score = sum(query_topics.get(str(topic), 0) * weight for topic, weight in doc_metadata['topics'].items())
#         doc_entity_score = len(set(ent[0] for ent in query_entities) & set(ent[0] for ent in doc_metadata['entities']))

#         total_score = 0.5 * distances[0][i] - 2.0 * doc_topic_score + 2.0 * doc_entity_score

#         if countries and any(country.lower() in doc_metadata['document'].lower() for country in countries):
#             total_score *= 0.5

#         doc_scores.append((total_score, idx))

#     # Sort and return the top-k documents
#     doc_scores.sort(key=lambda x: x[0])
#     return [metadata[str(i)]['document'] for _, i in doc_scores[:top_k]]

# # Updated generate_response function to use GPT-3.5 via OpenAI API
# def generate_response(query, countries=None):
#     # Predefined responses for greetings and small talk
#     greeting_responses = [
#         "Hello! How can I assist you today?",
#         "Hi there! What can I do for you?",
#         "Greetings! How can I help you today?"
#     ]
    
#     thank_you_responses = [
#         "You're welcome! Is there anything else you need?",
#         "No problem! Happy to help.",
#         "Glad I could assist. Anything else?"
#     ]
    
#     farewell_responses = [
#         "Goodbye! Have a great day!",
#         "Take care! If you need anything, feel free to ask.",
#         "Bye! I'm here if you need more help."
#     ]

#     introduction_responses = [
#         "I am NeuroClima Bot, an intelligent assistant designed to help you access and understand climate change policy data.",
#         "Hello! I'm NeuroClima Bot. My purpose is to provide information and insights about climate change policies.",
#         "I'm NeuroClima Bot. I'm here to assist you with climate change policy data and answer your questions related to it."
#     ]
    
#     # Handling greetings
#     if query.lower() in ["hello", "hi", "hey", "greetings"]:
#         return random.choice(greeting_responses)
    
#     # Handling thank you
#     elif "thank" in query.lower():
#         return random.choice(thank_you_responses)
    
#     # Handling farewells
#     elif query.lower() in ["bye", "goodbye", "see you", "take care"]:
#         return random.choice(farewell_responses)
    
#     # Handling introduction and bot's purpose
#     elif any(keyword in query.lower() for keyword in ["who are you", "what are you", "your purpose", "what do you do"]):
#         return random.choice(introduction_responses)

#     # For all other queries, proceed with the document retrieval and generation
#     else:
#         start_time = time.time() 
#         # Retrieve and summarize documents
#         retrieved_docs = retrieve_documents(query, countries)
#         summaries = summarize_documents(retrieved_docs)
#         context = " ".join(summaries)
#         print(context)
#         # Ensure the context length does not exceed the model's max length
#         max_input_length = 1000  # Adjust based on your model's max token length
#         context = context[:max_input_length]
        
#         # Define a prompt template
#         prompt_template = PromptTemplate(
#             input_variables=["query", "context"], 
#             template="Question: {query}\n\nContext: {context}\n\nPlease provide a detailed and comprehensive answer, using the context:"
#         )
        
#         # Initialize the OpenAI GPT-3.5 LLM instance
#         gpt3_5_llm = OpenAI_GPT3_5_LLM("gpt-4o-mini")
        
#         # Initialize the LLMChain with the prompt template and GPT-3.5 LLM
#         llm_chain = LLMChain(prompt=prompt_template, llm=gpt3_5_llm)
        
#         # Generate the response using LangChain
#         response = llm_chain.run({"query": query, "context": context})
        
#         end_time = time.time()  # Record end time
        
#         # Calculate the duration
#         duration = end_time - start_time
        
#         # Log or display the duration
#         print(f"Response generation took {duration:.2f} seconds.")

#         print("response1")
#         print(response)
#         def trim_and_refine_response(response):
#             # Find the last full stop followed by a space or the end of the string
#             last_full_stop = response.rfind('.')
            
#             # If no full stop is found, return the response as is
#             if last_full_stop == -1:
#                 return response.strip()

#             # Trim the response up to the last full stop
#             trimmed_response = response[:last_full_stop + 1].strip()

#             # Check the character before the last full stop in the trimmed response
#             char_before_full_stop = trimmed_response[last_full_stop - 1] if last_full_stop > 0 else ''

#             # If the character is a digit, check if it's part of a decimal number
#             if char_before_full_stop.isdigit():
#                 # Find the start of the number sequence with a possible decimal point before the last full stop
#                 number_start = re.search(r'\d+\.\d*$', trimmed_response[:last_full_stop])
#                 if number_start:
#                     # Remove the decimal number and the full stop
#                     return trimmed_response[:number_start.start()].strip()

#             # Return the trimmed response as is if it's a letter or any other non-digit character
#             return trimmed_response.strip()
#         refined_response = trim_and_refine_response(response)
#         print("trim_and_refine_response")
#         print(refined_response)
#         return refined_response

# # Example usage
# query = "What are Finland's policies related to managing chemical risks and hazardous substances?"
# country = ["Finland"]
# response = generate_response(query, country)
# print("Response:")
# print(response)

import torch
from transformers import AutoTokenizer, AutoModel, T5Tokenizer, T5ForConditionalGeneration, pipeline
import faiss
import numpy as np
import spacy
import os
import json
import openai
import random
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
import time
import re
from elasticsearch import Elasticsearch, exceptions
from elasticsearch.helpers import bulk
# from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
# from RAG.models.chunking_models import extract_entities, get_topic_distribution, summarize_documents
# from RAG.models.llm_models import (
#     retriever_model,
#     retriever_tokenizer,
#     summarizer_model,
#     summarizer_tokenizer,
# )
from transformers import AutoTokenizer, AutoModel, T5Tokenizer, T5ForConditionalGeneration

# Load Retriever Model and Tokenizer
retriever_model_name = "bert-base-uncased"
retriever_tokenizer = AutoTokenizer.from_pretrained(retriever_model_name)
retriever_model = AutoModel.from_pretrained(retriever_model_name)

# Load Summarizer Model and Tokenizer
summarizer_model_name = "t5-small"
summarizer_tokenizer = T5Tokenizer.from_pretrained(summarizer_model_name)
summarizer_model = T5ForConditionalGeneration.from_pretrained(summarizer_model_name)

def summarize_documents(docs, max_length=150):
    summaries = []
    for doc in docs:
        input_text = "summarize: " + doc
        inputs = summarizer_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = summarizer_model.generate(inputs, max_length=max_length, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
    return summaries

# Define a custom LLM class for OpenAI GPT-3.5
def OpenAI_GPT3_5_LLM(model, max_tokens=100, temperature=0.7, top_p=1):
    # Return a LangChain OpenAI LLM instance configured with the provided parameters
    return OpenAI(
        model_name=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p
    )

# Load the FAISS index
index_with_ids = faiss.read_index('saved_data/retriever_index_with_ids.faiss')

# Load environment variables from .env file
load_dotenv()

# Set the API key using the environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Connect to Elasticsearch
es = Elasticsearch("http://195.148.31.180:9200")  # Replace with your actual Elasticsearch endpoint

# Define the index name
INDEX_NAME = "test123"

# Function to retrieve documents based on the query from Elasticsearch
from elasticsearch import exceptions

def retrieve_documents(query, countries=None, top_k=4):
    # Tokenize and encode the query using the retriever model
    inputs = retriever_tokenizer(query, return_tensors="pt", truncation=True, padding=True)
    outputs = retriever_model(**inputs)
    query_embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()

    # Search the FAISS index using the query embedding
    distances, indices = index_with_ids.search(query_embedding, top_k)

    # Convert the query embedding to a list for Elasticsearch
    query_embedding_list = query_embedding.tolist()[0]

    # Construct the Elasticsearch query
    es_query = {
        "query": {
            "script_score": {
                "query": {
                    "match_all": {}
                },
                "script": {
                    "source": "dotProduct(params.query_vector, 'embedding') + 1.0",  # Add a small constant to the score
                    "params": {
                        "query_vector": query_embedding_list
                    }
                }
            }
        },
        "size": top_k
    }

    try:
        # Execute the Elasticsearch search
        es_results = es.search(index=INDEX_NAME, body=es_query)
        doc_scores = []

        # Iterate through Elasticsearch results to score and rank the retrieved documents
        for hit in es_results['hits']['hits']:
            doc_metadata = hit['_source']
            doc_id = hit['_id']  # Document ID from Elasticsearch

            # Calculate topic relevance score
            doc_topic_score = sum(doc_metadata['topics'].get(topic, 0) for topic in doc_metadata.get('topics', {}))

            # Calculate entity relevance score
            query_entities = set(ent[0] for ent in extract_entities(query))
            doc_entities = set(ent[0] for ent in doc_metadata.get('entities', []))
            doc_entity_score = len(query_entities & doc_entities)

            # Total score combining Elasticsearch score, FAISS distance, and topic/entity scores
            faiss_score = -1.0 * hit['_score']  # Using Elasticsearch '_score'
            total_score = faiss_score + 2.0 * doc_entity_score + doc_topic_score  # Modify the scoring logic as necessary
            doc_scores.append((total_score, doc_id))

        # Sort documents by their total score in ascending order (lower score is better)
        doc_scores.sort(key=lambda x: x[0])

        # Return the top-k documents based on sorted scores
        return [es_results['hits']['hits'][i]['_source']['document'] for _, i in doc_scores[:top_k]]

    except exceptions.ApiError as e:
        print(f"Elasticsearch search failed: {str(e)}")
        return []


# Updated generate_response function to use GPT-3.5 via OpenAI API
def generate_response(query, countries=None):
    # Predefined responses for greetings and small talk
    greeting_responses = [
        "Hello! How can I assist you today?",
        "Hi there! What can I do for you?",
        "Greetings! How can I help you today?"
    ]
    
    thank_you_responses = [
        "You're welcome! Is there anything else you need?",
        "No problem! Happy to help.",
        "Glad I could assist. Anything else?"
    ]
    
    farewell_responses = [
        "Goodbye! Have a great day!",
        "Take care! If you need anything, feel free to ask.",
        "Bye! I'm here if you need more help."
    ]

    introduction_responses = [
        "I am NeuroClima Bot, an intelligent assistant designed to help you access and understand climate change policy data.",
        "Hello! I'm NeuroClima Bot. My purpose is to provide information and insights about climate change policies.",
        "I'm NeuroClima Bot. I'm here to assist you with climate change policy data and answer your questions related to it."
    ]
    
    # Handling greetings
    if query.lower() in ["hello", "hi", "hey", "greetings"]:
        return random.choice(greeting_responses)
    
    # Handling thank you
    elif "thank" in query.lower():
        return random.choice(thank_you_responses)
    
    # Handling farewells
    elif query.lower() in ["bye", "goodbye", "see you", "take care"]:
        return random.choice(farewell_responses)
    
    # Handling introduction and bot's purpose
    elif any(keyword in query.lower() for keyword in ["who are you", "what are you", "your purpose", "what do you do"]):
        return random.choice(introduction_responses)

    # For all other queries, proceed with the document retrieval and generation
    else:
        start_time = time.time() 
        # Retrieve and summarize documents
        retrieved_docs = retrieve_documents(query, countries)
        summaries = summarize_documents(retrieved_docs)
        context = " ".join(summaries)
        print(context)
        # Ensure the context length does not exceed the model's max length
        max_input_length = 1000  # Adjust based on your model's max token length
        context = context[:max_input_length]
        
        # Define a prompt template
        prompt_template = PromptTemplate(
            input_variables=["query", "context"], 
            template="Question: {query}\n\nContext: {context}\n\nPlease provide a detailed and comprehensive answer, using the context:"
        )
        
        # Initialize the OpenAI GPT-3.5 LLM instance
        gpt3_5_llm = OpenAI_GPT3_5_LLM("gpt-4o-mini")
        
        # Initialize the LLMChain with the prompt template and GPT-3.5 LLM
        llm_chain = LLMChain(prompt=prompt_template, llm=gpt3_5_llm)
        
        # Generate the response using LangChain
        response = llm_chain.run({"query": query, "context": context})
        
        end_time = time.time()  # Record end time
        
        # Calculate the duration
        duration = end_time - start_time
        
        # Log or display the duration
        print(f"Response generation took {duration:.2f} seconds.")

        print("response1")
        print(response)

        def trim_and_refine_response(response):
            # Find the last full stop followed by a space or the end of the string
            last_full_stop = response.rfind('.')
            
            # If no full stop is found, return the response as is
            if last_full_stop == -1:
                return response.strip()

            # Trim the response up to the last full stop
            trimmed_response = response[:last_full_stop + 1].strip()

            # Check the character before the last full stop in the trimmed response
            char_before_full_stop = trimmed_response[last_full_stop - 1] if last_full_stop > 0 else ''

            # If the character is a digit, check if it's part of a decimal number
            if char_before_full_stop.isdigit():
                # Find the start of the number sequence with a possible decimal point before the last full stop
                number_start = re.search(r'\d+\.\d*$', trimmed_response[:last_full_stop])
                if number_start:
                    # Remove the decimal number and the full stop
                    return trimmed_response[:number_start.start()].strip()

            # Return the trimmed response as is if it's a letter or any other non-digit character
            return trimmed_response.strip()

        refined_response = trim_and_refine_response(response)
        print("trim_and_refine_response")
        print(refined_response)
        return refined_response

# Example usage
query = "What are Finland's policies related to managing chemical risks and hazardous substances?"
country = ["Finland"]
response = generate_response(query, country)
print("Response:")
print(response)
