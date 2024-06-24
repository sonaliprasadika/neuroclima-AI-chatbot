from flask import jsonify, request, make_response, json
from flask_restful import Resource
from data_model.db import db
import random

class StoreResource(Resource):
    def __init__(self):
        self.collection = db.weather_responses

    def post(self):
        data = request.get_json()
        if not data:
            return make_response(json.dumps({"error": "No data provided"}), 400)
        
        response_type = data.get('type')
        responses = data.get('responses')

        if not response_type or not responses:
            return make_response(json.dumps({"error": "Invalid data provided"}), 400)

        self.collection.insert_one({"type": response_type, "responses": responses})
        return  make_response(json.dumps({"message": "Responses stored successfully"}), 201)
