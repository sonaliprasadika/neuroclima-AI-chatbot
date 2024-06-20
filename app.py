from flask import Flask, render_template
from flask_restful import Api
from data_model.weather_responses import WeatherResponses
from routes import StoreResponse, GetResponse

app = Flask(__name__)
api = Api(app)

# Initialize the WeatherResponses class
weather_responses = WeatherResponses()

# Add routes to the API
api.add_resource(StoreResponse, '/store_responses')
api.add_resource(GetResponse, '/get_response')

@app.route("/")
def index():
    return render_template('chat.html')

if __name__ == '__main__':
    app.run(debug=True)
