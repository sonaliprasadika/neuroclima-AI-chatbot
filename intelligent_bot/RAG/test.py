import openai

# Replace 'your-api-key' with your actual OpenAI API key
openai.api_key = "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"

try:
    # Make a simple API call using the chat endpoint for gpt-3.5-turbo
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Use the chat model
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say something nice about OpenAI."}
        ],
        max_tokens=10
    )
    
    # Print the output from the API
    print("API Key is working! Response:")
    print(response['choices'][0]['message']['content'].strip())

except openai.error.AuthenticationError:
    print("Invalid API key! Please check your key and try again.")

except openai.error.OpenAIError as e:
    print(f"An error occurred: {str(e)}")
