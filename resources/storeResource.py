from flask import jsonify, request, make_response, json
from flask_restful import Resource
from data_model.elastic_search import es
import random

class StoreResource(Resource):
    def __init__(self):
        self.index = "weather_responses"

    def post(self):
        data = request.get_json()
        if not data:
            return make_response(json.dumps({"error": "No data provided"}), 400)
        
        response_type = data.get('type')
        responses = data.get('responses')

        if not response_type or not responses:
            return make_response(json.dumps({"error": "Invalid data provided"}), 400)

        doc = {"type": response_type, "responses": responses}

        try:
            res = es.index(index=self.index, body=doc)
            return make_response(json.dumps({"message": "Responses stored successfully", "id": res['_id']}), 201)
        except Exception as e:
            return make_response(json.dumps({"error": str(e)}), 500)
