from flask import jsonify, request
from flask_restful import Resource
from data_model.weather_responses import WeatherResponses

weather_responses = WeatherResponses()

class StoreResponse(Resource):
    def post(self):
        data = request.json
        if weather_responses.insert_responses(data):
            return jsonify({"message": "Responses stored successfully"}), 201
        return jsonify({"error": "Invalid data provided"}), 400

class GetResponse(Resource):
    def post(self):
        data = request.json
        response_type = data.get("type")
        if response_type:
            response = weather_responses.get_response(response_type)
            return jsonify({"response": response})
        return jsonify({"error": "Please provide a valid response type"}), 400
