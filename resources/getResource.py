<<<<<<< HEAD
from flask import request, make_response, json
from flask_restful import Resource
from data_model.db import db
import random

class GetResource(Resource):
    def __init__(self):
        self.collection = db.weather_responses
        
    def post(self):
        data = request.json
        response_type = data.get("type")
        if response_type:
            doc = self.collection.find_one({"type": response_type})
            if doc:
                response = random.choice(doc['responses'])
                return make_response(json.dumps(response), 200)
            else:
=======
from flask import request, make_response, json
from flask_restful import Resource
from data_model.db import db
import random

class GetResource(Resource):
    def __init__(self):
        self.collection = db.weather_responses
        
    def post(self):
        data = request.json
        response_type = data.get("type")
        if response_type:
            doc = self.collection.find_one({"type": response_type})
            if doc:
                response = random.choice(doc['responses'])
                return make_response(json.dumps(response), 200)
            else:
>>>>>>> 6d625fb (added req file and change view)
                return make_response(json.dumps({"error": "Please provide a valid response type"}), 400)