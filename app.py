<<<<<<< HEAD
from flask import Flask, render_template
from flask_restful import Api
from resources.storeResource import StoreResource
from resources.getResource import GetResource

app = Flask(__name__)
api = Api(app)

# Add routes to the API
api.add_resource(StoreResource, '/store')
api.add_resource(GetResource, '/get')

@app.route("/")
def index():
    return render_template('chat.html')

if __name__ == '__main__':
    app.run(debug=True)
=======
from flask import Flask, render_template
from flask_restful import Api
from resources.storeResource import StoreResource
from resources.getResource import GetResource

app = Flask(__name__)
api = Api(app)

# Add routes to the API
api.add_resource(StoreResource, '/store')
api.add_resource(GetResource, '/get')

@app.route("/")
def index():
    return render_template('chat.html')

if __name__ == '__main__':
    app.run(port=3000)
>>>>>>> 6d625fb (added req file and change view)
