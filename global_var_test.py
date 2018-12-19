from flask import Flask
from flask.json import jsonify
app = Flask(__name__)

cache = {}

@app.route("/create")
def create():
    cache['foo'] = 0
    return jsonify(cache['foo'])

@app.route("/increment")
def increment():
    cache['foo'] = cache['foo'] + 1
    return jsonify(cache['foo'])

@app.route("/read")
def read():
    return jsonify(cache['foo'])

if __name__ == '__main__':
    app.run()