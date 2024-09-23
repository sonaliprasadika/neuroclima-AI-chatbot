from flask import request, make_response, jsonify
from flask_restful import Resource
import logging
from RAG.lanchain_rag import retrieve_documents, generate_response

class GetResource(Resource):
    def __init__(self):
        # No need to specify an Elasticsearch index anymore
        pass

    def post(self):
        data = request.json
        query = data.get("query")
        countries = data.get("countries")  # Assuming countries could be passed in the request as well
        if query:
            try:
                # Log the incoming query
                logging.info(f"Received query: {query}")

                # Use the RAG retriever to get relevant documents
                retrieved_docs = retrieve_documents(query, countries)
                
                if not retrieved_docs:
                    return make_response(jsonify({"error": "No relevant documents found"}), 404)
                
                # Generate the final response using the RAG generator
                response = generate_response(query, countries)
                print("response:")
                print(response)
                logging.info(f"Generated result: {response}")

                # Prepare the response data
                response_data = {
                    "response": response
                }

                return make_response(jsonify(response_data), 200)
            except Exception as e:
                logging.error(f"Error processing query: {str(e)}")
                return make_response(jsonify({"error": str(e)}), 500)
        else:
            return make_response(jsonify({"error": "No query provided"}), 400)
