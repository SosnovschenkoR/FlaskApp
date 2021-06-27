import os
from datetime import time

from werkzeug.contrib.cache import SimpleCache
from flask import Flask
from flask.json import jsonify


app = Flask(__name__)

cache = SimpleCache()

@app.route("/create")
def create():
    #cache.set('foo', 0)
    return jsonify(cache.get('foo'))

@app.route("/increment")
def increment():
    cache.set('foo', cache.get('foo')+1)
    return jsonify(cache.get('foo'))

@app.route("/read")
def read():
    return jsonify(cache.get('foo'))

if __name__ == '__main__':
    cache.set('foo', 0)
    def test_job():
        while True:
            print("Run recurring task")
            time.sleep(3)

    app.run(host='127.0.0.1', debug=True, threaded=True, use_reloader=False)