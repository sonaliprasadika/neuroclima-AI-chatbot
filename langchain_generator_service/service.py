import torch
import faiss
import numpy as np
import spacy
import os
import json
import openai
import random
import requests
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
import time
import re
from models.chunking_models import summarize_documents
from models.llm_models import (
    summarizer_model,
    summarizer_tokenizer,
)

# Load environment variables from .env file
load_dotenv()

# Set the API key using the environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define a custom LLM class for OpenAI GPT-3.5
def OpenAI_GPT3_5_LLM(model, max_tokens=100, temperature=0.7, top_p=1):
    # Return a LangChain OpenAI LLM instance configured with the provided parameters
    return OpenAI(
        model_name=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p
    )

def retrieve_documents(query, countries):
    url = "http://localhost:5002/retrieve"
    payload = {
        "query": query,
        "countries": countries
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise an error for bad status codes
        return response.json()  # Assuming the response contains JSON data
    except requests.exceptions.RequestException as e:
        print(f"Error retrieving documents: {e}")
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
        return refined_response