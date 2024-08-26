import pandas as pd
import openai
import os
from dotenv import load_dotenv
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import T5Tokenizer, T5ForConditionalGeneration
import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

# Load environment variables from .env file
load_dotenv()

# Set the API key using the environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load the dataset
df = pd.read_csv('../../dataset/summery_training.csv', encoding='latin1')

# Initialize an empty list to store summaries
summaries = []

# Define a function to summarize the policy descriptions using GPT-4o-mini
def summarize_text(text):
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": (
                f"Summarize the following policy description. Focus on the key points, "
                f"ensure the summary is clear, and keep it within a length suitable for "
                f"include numbers, dates and every key word in the policy description:\n\n{text}"
            )}
        ],
        max_tokens=150  # Increase max tokens for a more detailed summary
    )
    return response['choices'][0]['message']['content'].strip()

# Loop through each policy description and summarize it
for policy in df['policy_description']:
    summary = summarize_text(policy)
    summaries.append(summary)

# Add the summaries to the dataframe
df['summary'] = summaries

# Save the dataframe with summaries to a new CSV file
df.to_csv('summarized_policies.csv', index=False)

# Load the local summarizer model (consider switching to a larger model like t5-large)
summarizer_model_name = "t5-large"  # Switch to a larger model for better summaries
summarizer_tokenizer = T5Tokenizer.from_pretrained(summarizer_model_name)
summarizer_model = T5ForConditionalGeneration.from_pretrained(summarizer_model_name)

# Load the dataset with original documents and GPT-4o-mini summaries
df = pd.read_csv('summarized_policies.csv', encoding='latin1')

# Extract original policy descriptions
original_docs = df['policy_description'].tolist()

# Extract GPT-4o-mini summaries
gpt_summaries = df['summary'].tolist()

# Function to summarize documents using your local model
def summarize_documents(docs, max_length=200):  # Increase max_length for more detailed summaries
    summaries = []
    for doc in docs:
        input_text = "summarize: " + doc
        inputs = summarizer_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = summarizer_model.generate(
            inputs,
            max_length=max_length,
            min_length=100,  # Increase min_length for more detailed summaries
            length_penalty=1.5,  # Adjust length penalty to encourage longer summaries
            num_beams=6,  # Increase num_beams for better quality
            early_stopping=True
        )
        summary = summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
    return summaries

# Generate summaries using your local model
local_summaries = summarize_documents(original_docs)

# Function to calculate cosine similarity between two lists of summaries
def calculate_cosine_similarity(list1, list2):
    vectorizer = TfidfVectorizer().fit_transform(list1 + list2)
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors[:len(list1)], vectors[len(list1):])
    return cosine_sim.diagonal().mean()

# Calculate and print cosine similarity between GPT-4o-mini summaries and local model summaries
cosine_sim = calculate_cosine_similarity(gpt_summaries, local_summaries)
print(f'Cosine Similarity between GPT-4o-mini and Local Model Summaries: {cosine_sim:.4f}')

# Calculate and print the Mean Squared Error (MSE) between GPT-4o-mini and Local Model summaries length
mse_length = mean_squared_error([len(s) for s in gpt_summaries], [len(s) for s in local_summaries])
print(f'Mean Squared Error of summary lengths: {mse_length:.4f}')

# # Function to calculate BLEU score
# def calculate_bleu_score(reference_summaries, generated_summaries):
#     bleu_scores = []
#     for reference, generated in zip(reference_summaries, generated_summaries):
#         reference_tokens = [reference.split()]
#         generated_tokens = generated.split()
#         score = sentence_bleu(reference_tokens, generated_tokens)
#         bleu_scores.append(score)
#     return sum(bleu_scores) / len(bleu_scores)

# # Calculate and print BLEU score
# bleu_score = calculate_bleu_score(gpt_summaries, local_summaries)
# print(f'BLEU Score between GPT-4o-mini and Local Model Summaries: {bleu_score:.4f}')

# Calculate ROUGE scores
def calculate_rouge_scores(gpt_summaries, local_summaries):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    
    for gpt_summary, local_summary in zip(gpt_summaries, local_summaries):
        score = scorer.score(local_summary, gpt_summary)
        for key in scores.keys():
            scores[key].append(score[key].fmeasure)
    
    avg_scores = {key: sum(values) / len(values) for key, values in scores.items()}
    return avg_scores

# Calculate and print ROUGE scores between GPT-4o-mini summaries and local model summaries
rouge_scores = calculate_rouge_scores(gpt_summaries, local_summaries)
print(f'ROUGE Scores:')
for metric, score in rouge_scores.items():
    print(f'{metric.upper()}: {score:.4f}')
    
# Combine results into a DataFrame for manual inspection
comparison_df = pd.DataFrame({
    'Original Document': original_docs,
    'GPT-4o-mini Summary': gpt_summaries,
    'Local Model Summary': local_summaries
})

# Save the comparison to a new CSV file
comparison_df.to_csv('summary_comparison.csv', index=False)

# Print out the comparison dataframe for manual inspection
print(comparison_df.head())
