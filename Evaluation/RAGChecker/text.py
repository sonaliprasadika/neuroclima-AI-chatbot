from ragchecker import RAGResults, RAGChecker
from ragchecker.metrics import all_metrics

# Define the custom function for invoking your own LLM
def my_llm_api_func(prompts):
    """
    Placeholder for LLM API logic.
    Replace this function with your own logic to fetch results from your LLM.
    """
    response_list = [f"Dummy response for: {prompt}" for prompt in prompts]  # Replace with real logic
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
