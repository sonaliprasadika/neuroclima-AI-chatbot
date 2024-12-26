from flask import Flask, jsonify, request

app = Flask(__name__)

# Mock generate endpoint
@app.route('/generate', methods=['POST'])
def generate():
    # Sample data: returning a mock generated response based on the query
    data = request.get_json()
    query = data.get('query', '')

    # Mocked responses for different queries
    if query == "What's the longest river in the world?":
        return jsonify({"generated_response": "The longest river in the world is the Nile, stretching approximately 6,650 kilometers (4,130 miles) through northeastern Africa."})
    elif query == "What does the democratic republic of Congo flag represent?":
        return jsonify({"generated_response": "The flag of the Democratic Republic of the Congo represents blue for peace, red for the blood of the country's martyrs, yellow for the country's wealth, and a star for a radiant future for the country."})
    else:
        return jsonify({"generated_response": "This is a mock response for: " + query})

# Mock retrieve endpoint
@app.route('/retrieve', methods=['POST'])
def retrieve():
    # Sample data: returning a mock list of retrieved documents based on the query
    data = request.get_json()
    query = data.get('query', '')

    # Mocked context for different queries
    if query == "What's the longest river in the world?":
        context = [
            {"doc_id": "000", "text": "The Nile River has traditionally been considered the longest river in the world."},
            {"doc_id": "001", "text": "The Amazon River might be longer, depending on how its tributaries are counted."}
        ]
    elif query == "What does the democratic republic of Congo flag represent?":
        context = [
            {"doc_id": "000", "text": "The DRC flag includes blue for peace, red for martyrs' blood, yellow for wealth, and a star for the future."},
            {"doc_id": "001", "text": "The DRC flag was adopted in 2006, symbolizing hope, unity, and the country's rich resources."}
        ]
    else:
        context = [{"doc_id": "000", "text": "This is a mock document for: " + query}]
    
    return jsonify({"retrieved_context": context})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
