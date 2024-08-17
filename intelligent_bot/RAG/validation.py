import pandas as pd
import re

# Load the CSV file
csv_file_path = "../dataset/validation.csv"  
try:
    df = pd.read_csv(csv_file_path, encoding='ISO-8859-1')
except UnicodeDecodeError:
    df = pd.read_csv(csv_file_path, encoding='utf-8')

# Initialize a list to store results and counters for the total score
results = []
total_percentage_score_sum = 0
number_of_queries = 0

# Iterate through each row in the DataFrame
for index, row in df.iterrows():
    # Get the response and entities from the row
    text = row['Response'].lower()  # Convert response text to lowercase
    search_terms = re.findall(r'\b\w+%?\b', row['Entities'].lower())  # Extract entities and convert to lowercase
    
    # Initialize a list for found entities
    present_entities = []
    
    # Count occurrences of each entity in the response text
    for term in search_terms:
        if re.search(r'\b' + re.escape(term) + r'\b', text):
            present_entities.append(term)
    
    # Calculate the percentage score for the current query
    query_total_target_types = len(search_terms)
    query_present_count = len(present_entities)
    query_percentage_score = (query_present_count / query_total_target_types) * 100 if query_total_target_types > 0 else 0
    
    # Append results for this query
    results.append({
        'Query': row['Queries'],
        'Present Entities': present_entities,
        'Percentage Score': query_percentage_score
    })
    
    # Accumulate the sum of percentage scores and count the number of queries
    total_percentage_score_sum += query_percentage_score
    number_of_queries += 1

# Calculate the average percentage score across all queries
average_percentage_score = (total_percentage_score_sum / number_of_queries) if number_of_queries > 0 else 0

# Display results for each query
for result in results:
    print(f"Query: {result['Query']}")
    print(f"Present Entities: {result['Present Entities']}")
    print(f"Percentage Score: {result['Percentage Score']:.2f}%")
    print("----" * 10)

# Display the average percentage score
print(f"Average Percentage Score across all queries: {average_percentage_score:.2f}%")
