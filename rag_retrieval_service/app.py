from flask import Flask, request, jsonify
from service import retrieve_documents

app = Flask(__name__)

@app.route('/retrieve', methods=['POST'])
def retrieve():
    data = request.json
    query = data.get("query")
    countries = data.get("countries")
    response = retrieve_documents(query, countries)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(port=5002)
 