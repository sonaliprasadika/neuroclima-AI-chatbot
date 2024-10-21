from flask import Flask, request, jsonify
from retrieve_service import retrieve_documents, generate_response

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    query = data.get("query")
    countries = data.get("countries")
    response = generate_response(query, countries)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(port=5003)
