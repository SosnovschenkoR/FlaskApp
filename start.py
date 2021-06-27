import datetime
import json
import time

from flask import Flask, render_template, Response
from camera import VideoCamera

app = Flask(__name__)


def success_handle(output, status=200, mimetype='application/json'):
    return Response(output, status=status, mimetype=mimetype)


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


@app.route('/hello')
def hello_world():
    return 'Hello, World!'


def gen(camera):
    while True:
        frame = camera.get_frame()
        time.sleep(1.0)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/request_count')
def request_count():
    value = 'n/a'
    info = {"pers": str(value), "date": datetime.datetime.now()}

    def customconverter(o):
        if isinstance(o, datetime.datetime):
            return o.__str__()

    output = json.dumps(info, default=customconverter)
    return success_handle(output)


@app.route('/api', methods=['GET'])
def homepage():
    output = json.dumps({"api": '1.0'})
    return success_handle(output)


if __name__ == "__main__":
    app.run(host='127.0.0.1', debug=True, threaded=True)
