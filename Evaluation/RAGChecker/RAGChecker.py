from ragchecker import RAGResults, RAGChecker
from ragchecker.metrics import all_metrics
import openai

# Set your API key
openai.api_key = "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"


# initialize ragresults from json/dict
with open("examples/checking_inputs.json") as fp:
    rag_results = RAGResults.from_json(fp.read())

# set-up the evaluator
evaluator = RAGChecker(
    extractor_name="gpt-3.5-turbo",
    checker_name="gpt-3.5-turbo",
    batch_size_extractor=32,
    batch_size_checker=32
)

# evaluate results with selected metrics or certain groups, e.g., retriever_metrics, generator_metrics, all_metrics
evaluator.evaluate(rag_results, all_metrics)
print(rag_results)




# import openai

# # Set your API key
# openai.api_key = "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"

# # Test the API key
# try:
#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant."},
#             {"role": "user", "content": "Test the OpenAI API."},
#         ],
#         max_tokens=10,
#     )
#     print("API Key is working. Test response:", response["choices"][0]["message"]["content"].strip())

# except openai.AuthenticationError as auth_error:
#     print("Authentication Error: Please check your OpenAI API key.")
#     print(auth_error)
# except openai.OpenAIError as general_error:
#     print(f"OpenAI API Error: {general_error}")
# except Exception as e:
#     print(f"Unexpected Error: {e}")


# import os
# from ragchecker.integrations.llama_index import response_to_rag_results
# from ragchecker import RAGResults, RAGChecker
# from ragchecker.metrics import all_metrics

# # User query and ground truth answer
# user_query = "What is RAGChecker?"
# gt_answer = (
#     "RAGChecker is an advanced automatic evaluation framework designed to assess and "
#     "diagnose Retrieval-Augmented Generation (RAG) systems. It provides a comprehensive "
#     "suite of metrics and tools for in-depth analysis of RAG performance."
# )

# # Mock response object
# class ResponseObject:
#     def __init__(self, response, source_nodes):
#         self.response = response
#         self.source_nodes = source_nodes

# response_object = ResponseObject(
#     response=gt_answer,
#     source_nodes=[],  # Add source nodes here if available
# )

# # Convert to RAGChecker format
# rag_result = response_to_rag_results(
#     query=user_query,
#     gt_answer=gt_answer,
#     response_object=response_object,
# )

# # Create RAGResults object
# rag_results = RAGResults.from_dict({"results": [rag_result]})

# # Initialize RAGChecker
# evaluator = RAGChecker(
#     extractor_name="gpt-4o-mini",
#     checker_name="gpt-4o-mini",
#     batch_size_extractor=32,
#     batch_size_checker=32,
# )

# # Evaluate and print results
# try:
#     evaluator.evaluate(rag_results, all_metrics)
#     print("RAG Results Evaluation Complete:")
#     print(rag_results)
# except Exception as e:
#     print(f"Evaluation Error: {e}")



