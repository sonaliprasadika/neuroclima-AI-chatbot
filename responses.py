from flask import Flask, request, jsonify, render_template
from pymongo import MongoClient
import random

app = Flask(__name__)

# Setup MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['weather_db']

class WeatherResponses:
    def __init__(self, db):
        self.db = db
        self.collection = self.db.weather_responses

    def insert_responses(self, data):
        response_type = data.get('type')
        responses = data.get('responses')
        if response_type and responses:
            # Insert a new response into the collection
            self.collection.insert_one({"type": response_type, "responses": responses})
            return True
        return False

    def get_response(self, response_type):
        doc = self.collection.find_one({"type": response_type})
        if doc:
            return random.choice(doc['responses'])
        else:
            return "No responses found for this type."

# Initialize the WeatherResponses class
weather_responses = WeatherResponses(db)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/store_responses", methods=["POST"])
def store_responses():
    data = request.json
    if weather_responses.insert_responses(data):
        return jsonify({"message": "Responses stored successfully"}), 201
    return jsonify({"error": "Invalid data provided"}), 400

@app.route("/get_response", methods=["POST"])
def get_response():
    data = request.json
    response_type = data.get("type")
    if response_type:
        response = weather_responses.get_response(response_type)
        return jsonify({"response": response})
    return jsonify({"error": "Please provide a valid response type"}), 400

if __name__ == '__main__':
    app.run(debug=True)
