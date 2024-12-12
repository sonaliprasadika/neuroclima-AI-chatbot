from deepeval import evaluate
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase
import openai
import os
from dotenv import load_dotenv
import pandas as pd

# Load environment variables from the .env file
load_dotenv()

# Ensure the API key is loaded
openai.api_key = os.getenv("OPENAI_API_KEY")
if openai.api_key is None:
    raise ValueError("API key not found. Please check your .env file.")

# Load the data from the CSV file
df = pd.read_csv('deepeval_evaluation.csv', encoding='latin1')

# Extract the relevant columns
inputs = df['input'].tolist()
print(inputs)
actual_outputs = df['actual_output'].tolist()
expected_outputs = df['expected_output'].tolist()
retrieval_context = df['retrieval_context'].apply(lambda x: x.split(';') if isinstance(x, str) else [x]).tolist()
provided_context = df['provided context'].apply(lambda x: x.split(';') if isinstance(x, str) else [x]).tolist()

n=3
# Initialize the metric
metric = ContextualPrecisionMetric(
    threshold=0.7,
    model="gpt-4",  # Ensure you have access to this model
    include_reason=True
)

# Create a list to hold test cases
test_cases = []

# Loop through the data to create 10 test cases
for i in range(n):
    test_case = LLMTestCase(
        input=inputs[i],
        actual_output=actual_outputs[i],
        expected_output=expected_outputs[i],
        retrieval_context=retrieval_context[i],
        context=provided_context[i]
    )
    test_cases.append(test_case)

# Evaluate test cases in bulk
evaluate(test_cases, [metric])

# Optionally, print the scores and reasons
for test_case in test_cases:
    if hasattr(test_case, 'metrics') and test_case.metrics:  # Ensure metrics are available
        print(f"Test case input: {test_case.input}")
        print(f"Score: {test_case.metrics[0].score}")
        print(f"Reason: {test_case.metrics[0].reason}")
    else:
        print(f"Metrics not available for input: {test_case.input}")
    print("="*50)

