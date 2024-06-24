from pymongo import MongoClient

client = MongoClient('mongodb://128.214.253.165:27018/')
db = client['weather_db']
