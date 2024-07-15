from flask import request, make_response, jsonify
from flask_restful import Resource
from data_model.elastic_search import es
from RAG.rag_bot import combined_chain
import logging

class GetResource(Resource):
    def __init__(self):
        self.index = "policy_descriptions"

    def post(self):
        data = request.json
        query = data.get("query")
        if query:
            try:
                # Log the incoming query
                logging.info(f"Received query: {query}")

                # Search the documents in Elasticsearch
                retrieved_docs = self.search_documents(query)
                combined_context = " ".join([doc["policy_description"] for doc in retrieved_docs])
                docs = [doc["policy_description"] for doc in retrieved_docs]
                inputs = {
                    "context": combined_context,
                    "query": query,
                    "docs": docs
                }

                # Run the combined chain to get the response
                result = combined_chain.run(inputs)
                logging.info(f"Generated result: {result}")

                # Prepare the response data
                response_data = {
                    "topics": result["topics"],
                    "entities": result["entities"],
                    "summary": result.get("summary", ""),
                    "response": result["response"]
                }

                return make_response(jsonify(response_data), 200)
            except Exception as e:
                logging.error(f"Error processing query: {str(e)}")
                return make_response(jsonify({"error": str(e)}), 500)
        else:
            return make_response(jsonify({"error": "No query provided"}), 400)

    def search_documents(self, query, index_name="policy_descriptions", size=3):
        search_query = {
            "query": {
                "match": {
                    "policy_description": query
                }
            },
            "size": size
        }
        response = es.search(index=index_name, body=search_query)
        return [hit["_source"] for hit in response["hits"]["hits"]]
