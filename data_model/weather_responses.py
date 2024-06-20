from data_model.db import db
import random

class WeatherResponses:
    def __init__(self):
        self.collection = db.weather_responses

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
