from ragchecker import RAGResults, RAGChecker
from ragchecker.metrics import all_metrics
import requests

def my_llm_api_func(prompts):
    response_list = []
    for prompt in prompts:
        
        # Replace with your custom API endpoint and authentication
        api_url = "http://127.0.0.1:5002/generate"
        payload = {"query": prompt}
        
        response = requests.post(api_url, json=payload)
        print(response.json().get("generated_response", ""))
        response_list.append(response.json().get("generated_response", ""))
    print(response_list)
    return response_list

# Load RAGResults from JSON
with open("examples/checking_inputs.json") as fp:
    rag_results = RAGResults.from_json(fp.read())

# Initialize evaluator with your custom LLM API function
evaluator = RAGChecker(
    custom_llm_api_func=my_llm_api_func,
    batch_size_extractor=32,
    batch_size_checker=32
)

# Evaluate results
evaluator.evaluate(rag_results, all_metrics)

# Print the results
print(rag_results)