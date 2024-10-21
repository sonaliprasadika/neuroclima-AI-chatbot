from pymongo import MongoClient

client = MongoClient('mongodb://localhost:9005')
db = client['weather_db']
# collection = db['weather_responses']