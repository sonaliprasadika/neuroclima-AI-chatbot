from ragas import evaluate
from datasets import Dataset
import openai
import os
from dotenv import load_dotenv  # Import load_dotenv
import pandas as pd

# Load environment variables from the .env file
load_dotenv()  # This will load the key into the environment

# Make sure the API key is set in the environment
openai.api_key = os.getenv("OPENAI_API_KEY")

# Check if the key is loaded
if openai.api_key is None:
    raise ValueError("API key not found. Please check your .env file.")

# Load the data from the CSV file
df = pd.read_csv('deepeval_evaluation.csv', encoding='latin1')

# Extract the relevant columns
inputs = df['input'].tolist()
actual_outputs = df['actual_output'].tolist()
expected_outputs = df['expected_output'].tolist()
retrieval_context = df['retrieval_context'].apply(lambda x: x.split(';') if isinstance(x, str) else [x]).tolist()

# Convert the data into the format required by the Dataset object
eval_dataset = Dataset.from_dict({
    "question": inputs,
    "contexts": retrieval_context,
    "answer": actual_outputs,
    "ground_truths": retrieval_context,  # Assuming retrieval_context serves as ground truths
    "reference": expected_outputs
})

# Run the evaluation
results = evaluate(eval_dataset)
print(results)
