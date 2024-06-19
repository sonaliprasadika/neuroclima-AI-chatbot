from flask import Flask, render_template, request, jsonify
from rulebot import RuleBot

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg")
    if msg:
        bot = RuleBot()
        response = bot.match_reply(msg)
        return jsonify({"response": response})
    return jsonify({"response": "No message received"}), 400

if __name__ == '__main__':
    app.run(debug=True)