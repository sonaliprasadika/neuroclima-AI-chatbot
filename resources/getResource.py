from flask import request, make_response, json
from flask_restful import Resource
from data_model.elastic_search import es
import random

class GetResource(Resource):
    def __init__(self):
        self.index = "weather_responses"

    def post(self):
        data = request.json
        response_type = data.get("type")
        if response_type:
            query = {
                "query": {
                    "match": {
                        "type": response_type
                    }
                }
            }

            try:
                res = es.search(index=self.index, body=query)
                print(dir(es))
                hits = res['hits']['hits']
                if hits:
                    responses = hits[0]['_source']['responses']
                    response = random.choice(responses)
                    return make_response(json.dumps(response), 200)
                else:
                    return make_response(json.dumps({"error": "Please provide a valid response type"}), 400)
            except Exception as e:
                return make_response(json.dumps({"error": str(e)}), 500)
        else:
            return make_response(json.dumps({"error": "No response type provided"}), 400)