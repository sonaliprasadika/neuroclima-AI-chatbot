from flask import request, make_response, jsonify
from flask_restful import Resource
import logging
import requests

class GetResource(Resource):
    def post(self):
        data = request.json
        query = data.get("query")
        print(data)
        countries = data.get("countries")  # Assuming countries could be passed in the request as well
        if query:
            try:
                # Log the incoming query
                logging.info(f"Received query: {query}")

                # Send the query to the generator service (running on port 5003)
                # generator_url = "http://86.50.230.170:5003/generate"
                generator_url = "http://127.0.0.1:5003/generate"
                response = requests.post(generator_url, json={"query": query, "countries": countries})
                
                if response.status_code != 200:
                    return make_response(jsonify({"error": "No relevant documents found"}), 404)
                
                # Process the response from the generator service
                response_data = response.json()
                return make_response(jsonify(response_data), 200)
            except Exception as e:
                logging.error(f"Error processing query: {str(e)}")
                return make_response(jsonify({"error": str(e)}), 500)
        else:
            return make_response(jsonify({"error": "No query provided"}), 400)
