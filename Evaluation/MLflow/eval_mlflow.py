import os
from dotenv import load_dotenv
import pandas as pd
import mlflow
from mlflow.metrics.genai import faithfulness, relevance

# Load environment variables from the .env file
load_dotenv()

# Make sure the API key is set in the environment (not directly needed for MLflow but retained for completeness)
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    raise ValueError("API key not found. Please check your .env file.")

# Load the data from the CSV file
df = pd.read_csv('deepeval_evaluation.csv', encoding='latin1')

# Extract the relevant columns
inputs = df['input'].tolist()
actual_outputs = df['actual_output'].tolist()
expected_outputs = df['expected_output'].tolist()
retrieval_context = df['retrieval_context'].apply(lambda x: x.split(';') if isinstance(x, str) else [x]).tolist()

# Create a Pandas DataFrame for evaluation
eval_df = pd.DataFrame({
    "inputs": inputs,
    "outputs": actual_outputs,
    "ground_truth": expected_outputs,
    "context": retrieval_context,  # Rename retrieval_context to context
})

# Convert to an MLflow-compatible dataset
eval_dataset = mlflow.data.from_pandas(
    eval_df, predictions="outputs", targets="ground_truth"
)

# Define relevance and faithfulness metrics
relevance_metric = relevance(model="openai:/gpt-4")
faithfulness_metric = faithfulness(model="openai:/gpt-4")

print(relevance_metric)
print(faithfulness_metric)

with mlflow.start_run(run_name="RAG_Evaluation"):
    # Run evaluation using the default evaluator and additional metrics
    results = mlflow.evaluate(
        data=eval_dataset,
        model_type="question-answering",
        evaluators=["default"],  # Use default evaluator
        extra_metrics=[relevance_metric, faithfulness_metric],
    )

    # Extract the relevance and faithfulness scores from the results
    relevance_score = results.metrics.get("relevance_metric", None)
    faithfulness_score = results.metrics.get("faithfulness", None)

    # Print the relevance and faithfulness scores
    print(f"Relevance Score: {relevance_score}")
    print(f"Faithfulness Score: {faithfulness_score}")

    # Convert the evaluation results table to a DataFrame
    results_table = results.tables.get("eval_results_table")
    if results_table is not None:
        results_file_path = "rag_evaluation_results.csv"
        results_table.to_csv(results_file_path, index=False)
        print(f"Evaluation results saved to {results_file_path}")
    else:
        print("No results table available.")
