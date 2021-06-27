import os
from pathlib import Path

from flask import Flask, url_for, abort, render_template, json
from flask.json import jsonify
import configparser


app = Flask(__name__)
cache = {}

config_file = Path("config.json")
if config_file.is_file():
    config = configparser.ConfigParser()
    with open(config_file.name, 'r') as f:
        config = json.load(f)
    prototxt_path = config['DEFAULT']['PROTOTXT']
    model_path = config['DEFAULT']['MODEL']
    confidence_val = config['DEFAULT']['CONFIDENCE']
else:
    model_path = os.getenv('MODEL_PATH')
    prototxt_path = os.getenv('PROTOTXT_PATH')
    confidence_val = 0.2

cameras = config['CAMERAS']
if (len(cameras) > 0):
    exit(0)

# cameras = [
#     {
#         'id': 1,
#         'url': u'http://127.0.0.1:8081/video.mjpg',
#         'description': u'БК1-301'
#     },
#     {
#         'id': 2,
#         'url': u'http://127.0.0.1:8081/video.mjpg',
#         'description': u'БК1-302'
#     }
# ]

@app.route('/todo/api/v1.0/cameras', methods=['GET'])
def get_tasks():
    return jsonify({'cameras': cameras})


def make_public_task(camera):
    new_task = {}
    for field in camera:
        if field == 'id':
            new_task['uri'] = url_for('get_task', camera_id = camera['id'], _external = True)
            new_task['id'] = camera['id']
        else:
            new_task[field] = camera[field]
    return new_task


@app.route('/todo/api/v1.0/cameras/<int:camera_id>', methods = ['GET'])
def get_task(camera_id):
    camera = list(filter(lambda t: t['id'] == camera_id, cameras))
    if len(camera) == 0:
        abort(404)
    return jsonify( { 'camera': make_public_task(camera[0]) } )


@app.route('/')
def index():
    #return 'Start page<br>' + 'Camera count = ' + config['DEFAULT']['CAMERA_COUNT'];
    return render_template("home.html", )


# @app.route('/request_count')
# def index():
#     #return 'Start page<br>' + 'Camera count = ' + config['DEFAULT']['CAMERA_COUNT'];
#     return render_template("home.html", )


@app.route('/hello')
def hello_world():
    return 'Hello, World!'


@app.route("/create")
def create():
    cache['foo'] = 0
    return jsonify(cache['foo'])


@app.route("/increment", defaults={'path': ''})
def increment():
    cache['foo'] = cache['foo'] + 1
    return jsonify(cache['foo'])


@app.route("/read")
def read():
    return jsonify(cache['foo'])


def has_no_empty_params(rule):
    defaults = rule.defaults if rule.defaults is not None else ()
    arguments = rule.arguments if rule.arguments is not None else ()
    return len(defaults) >= len(arguments)


@app.route("/all-links")
def all_links():
    links = []
    for rule in app.url_map.iter_rules():
        #if len(rule.defaults) >= len(rule.arguments):
        if "GET" in rule.methods and has_no_empty_params(rule):
            url = url_for(rule.endpoint, **(rule.defaults or {}))
            links.append((url, rule.endpoint))
    return render_template("all_links.html", links=links)


if __name__ == '__main__':
    app.run()
