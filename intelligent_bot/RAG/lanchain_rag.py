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
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from RAG.models.chunking_models import extract_entities, get_topic_distribution, summarize_documents
from RAG.models.llm_models import (
    retriever_model,
    retriever_tokenizer,
    summarizer_model,
    summarizer_tokenizer,
)
# from models.chunking_models import extract_entities, get_topic_distribution, summarize_documents
# from models.llm_models import (
#     retriever_model,
#     retriever_tokenizer,
#     summarizer_model,
#     summarizer_tokenizer,
# )
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
import time
import re
# Load the FAISS index and metadata from files
index_with_ids = faiss.read_index('saved_data/retriever_index_with_ids.faiss')

with open('saved_data/metadata.json', 'r', encoding='utf-8') as f:
    metadata = json.load(f)

# Load environment variables from .env file
load_dotenv()

# Set the API key using the environment variable

# Define a custom LLM class for OpenAI GPT-3.5
def OpenAI_GPT3_5_LLM(model, max_tokens=150, temperature=0.7, top_p=1):
    # Return a LangChain OpenAI LLM instance configured with the provided parameters
    return ChatOpenAI(
        model=model,  # Use 'model' instead of 'model_name'
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        request_timeout=60,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

# Function to retrieve documents based on the query
def retrieve_documents(query, countries=None, top_k=4):
    # Tokenize and encode the query using the retriever model
    inputs = retriever_tokenizer(query, return_tensors="pt", truncation=True, padding=True)
    outputs = retriever_model(**inputs)
    query_embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()

    # Search the FAISS index using the query embedding
    distances, indices = index_with_ids.search(query_embedding, len(metadata))

    query_topics = get_topic_distribution(query)
    query_entities = extract_entities(query)

    # Score and rank the retrieved documents
    doc_scores = []
    for i in range(len(indices[0])):
        idx = indices[0][i]
        doc_metadata = metadata[str(idx)]

        if countries and not any(country.lower() in doc_metadata['document'].lower() for country in countries):
            continue

        doc_topic_score = sum(query_topics.get(str(topic), 0) * weight for topic, weight in doc_metadata['topics'].items())
        doc_entity_score = len(set(ent[0] for ent in query_entities) & set(ent[0] for ent in doc_metadata['entities']))

        total_score = 0.5 * distances[0][i] - 2.0 * doc_topic_score + 2.0 * doc_entity_score

        if countries and any(country.lower() in doc_metadata['document'].lower() for country in countries):
            total_score *= 0.5

        doc_scores.append((total_score, idx))

    # Sort and return the top-k documents
    doc_scores.sort(key=lambda x: x[0])
    return [metadata[str(i)]['document'] for _, i in doc_scores[:top_k]]

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
        "I'm NeuroClima Bot. My purpose is to provide information and insights about climate change policies.",
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

        # Ensure the context length does not exceed the model's max length
        max_input_length = 4096 - 500  # Adjust based on your model's max token length
        context = context[:max_input_length]

        print("context:")
        print(context)

        # Define a prompt template
        prompt_template = PromptTemplate(
            input_variables=["query", "context"], 
            template="Question: {query}\n\nContext: {context}\n\nAnswer:"
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

        # Refine response to stop at a complete sentence or list item
        def refine_response(response):
            # Find the last full stop in the response
            last_full_stop = response.rfind('. ')

            # If no full stop is found, return the response as is
            if last_full_stop == -1:
                return response

            # Check if the response ends with a number followed by a period
            if re.match(r'\d+\.', response[last_full_stop + 1:].strip()):
                # Find the end of the last list item
                next_full_stop = response.find('.', last_full_stop + 1)
                if next_full_stop != -1:
                    return response[:next_full_stop + 1]

            return response[:last_full_stop + 1]

        # Apply refinement to response
        refined_response = refine_response(response)

        return refined_response

# Example usage
query = "How is Norway addressing emissions from the waste sector?"
country = ["Norway"]
response = generate_response(query, country)
print("Response:")
print(response)
