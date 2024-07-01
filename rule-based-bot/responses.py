from pymongo import MongoClient
import random

class WeatherResponses:
    def __init__(self, db):
        self.db = db
        self.collection = self.db.weather_responses

    def insert_responses(self):
        responses = {
            "general_weather_queries": [
                "Today’s weather is sunny\n",
                "There’s a 60% chance of rain.\n"
            ],
            "weather_condition": [
                "The current temperature is 18°C.\n",
                "The wind speed today is 1 m/s, coming from the north.\n",
                "There is no storm expected today.\n"
            ],
            "future_forecasts": [
                "This weekend, the weather will be cloudy. Expect temperatures between 12°C and 16°C.\n",
                "Next week’s forecast shows temperatures will be high, ranging from 24°C to 26°C.\n"
            ],
            "no_match_intent": [
                "Please tell me more.\n", "Tell me more!\n", "I see. Can you elaborate?\n",
                "Interesting. Can you tell me more?\n", "I see. How do you think?\n", "Why?\n",
                "How do you think I feel when I say that? Why?\n"
            ]
        }
        
        # Insert responses into the collection
        for key, value in responses.items():
            self.collection.insert_one({"type": key, "responses": value})

    def general_weather_queries(self):
        doc = self.collection.find_one({"type": "general_weather_queries"})
        return random.choice(doc['responses'])

    def weather_condition(self):
        doc = self.collection.find_one({"type": "weather_condition"})
        return random.choice(doc['responses'])

    def future_forecasts(self):
        doc = self.collection.find_one({"type": "future_forecasts"})
        return random.choice(doc['responses'])

    def no_match_intent(self):
        doc = self.collection.find_one({"type": "no_match_intent"})
        return random.choice(doc['responses'])

# Setup MongoDB connection
client = MongoClient('mongodb://localhost:9005/')
db = client['weather_db']

# Initialize the WeatherResponses class
weather_responses = WeatherResponses(db)

# Insert the responses into MongoDB
weather_responses.insert_responses()

# Example usage
print(weather_responses.general_weather_queries())
print(weather_responses.weather_condition())
print(weather_responses.future_forecasts())
print(weather_responses.no_match_intent())
