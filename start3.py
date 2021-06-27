import os
import signal
import sys
import argparse
import datetime
import json
import time
import threading

import configparser
from pathlib import Path

import imutils as imutils
import numpy as np
import cv2

from flask import Flask, render_template, Response
from werkzeug.contrib.cache import SimpleCache
from camera2 import VideoCamera

print("Application starting...")

cache = SimpleCache()
thread = threading.Thread()

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=False, help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=False, help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2, help="minimum probability to filter weak detections")
#args = vars(ap.parse_args())

config = configparser.ConfigParser()

config_file = Path("config.json")
if config_file.is_file():
    config = configparser.ConfigParser()
    with open(config_file.name, 'r') as f:
        config = json.load(f)
    prototxt_path = config['DEFAULT']['PROTOTXT']
    model_path = config['DEFAULT']['MODEL']
    confidence_val = config['DEFAULT']['CONFIDENCE']
    cameras = config['CAMERAS']
else:
    # model_path = "../OpenCV/MobileNetSSD_deploy.caffemodel"
    # prototxt_path = "../OpenCV/MobileNetSSD_deploy.prototxt"
    model_path = os.getenv('MODEL_PATH')
    prototxt_path = os.getenv('PROTOTXT_PATH')
    confidence_val = 0.2
    cameras = [
        {
            'id': 1,
            'url': u'http://127.0.0.1:8081/video.mjpg',
            'description': u'БК1-301'
        },
        {
            'id': 2,
            'url': u'http://127.0.0.1:8081/video.mjpg',
            'description': u'БК1-302'
        }
    ]

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
#net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
#net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
#net2 = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

def people_counter_job(camera):
    global net2
    global thread
    obj_count = 0

    label = '{:%d.%m.%Y %H:%M:%S}'.format(datetime.datetime.now())
    # print('begin get_frame: ' + label)
    frame = camera.get_frame()
    if frame is None:
        print('Empty frame: ' + label)
        thread = threading.Timer(5.0, people_counter_job, (VideoCamera(),))
        thread.start()
        return

    frame = imutils.resize(frame, width=400)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    # pass the blob through the network and obtain the detections and
    # predictions
    net2.setInput(blob)
    detections = net2.forward()
    obj_count = 0
    now = datetime.datetime.now()
    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        # args["confidence"]
        if confidence > confidence_val:
            # extract the index of the class label from the
            # `detections`, then compute the (x, y)-coordinates of
            # the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] != "person":
                continue
            obj_count = obj_count + 1
    if (now.second > 0 and now.second <= 5):
        cache.set('global_obj_count', obj_count)
        cache.set('global_value_time', now)
    else:
        value = cache.get('global_obj_count')
        if obj_count > value:
            cache.set('global_obj_count', obj_count)
            cache.set('global_value_time', now)
    print('end get_frame: ' + label)
    thread = threading.Timer(5.0, people_counter_job, (VideoCamera(),))
    thread.start()

def people_counter_job2():
    global cache
    local_num = cache.get('global_obj_count')
    local_num += 1
    cache.set('global_obj_count', local_num)
    cache.set('global_value_time', datetime.datetime.now())
    print(local_num)

    global thread
    thread = threading.Timer(1.0, people_counter_job2, ())
    thread.start()

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    interrupt()

def interrupt():
    global thread
    thread.cancel()
    print("Thread exit")

def doBeforeStart():
    for camera in cameras:
        cache.set('global_obj_count' + str(camera['id']), 0)
        cache.set('global_value_time' + str(camera['id']), 0)
    # lock to control access to variable
    dataLock = threading.Lock()
    # thread handler

    global thread
    #thread = threading.Timer(1.0, people_counter_job, ())
    thread = threading.Timer(1.0, people_counter_job, (VideoCamera(),))
    thread.start()

def create_app():
    local_app = Flask(__name__)
    print("Run recurring task")
    # Initiate
    doBeforeStart()

    # When you kill Flask (SIGTERM), clear the trigger for the next thread
    signal.signal(signal.SIGINT, signal_handler)
    return local_app

app = create_app()
print("Application started")


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html', title="Э-316(1-й монтёрский)")


@app.route('/hello')
def hello_world():
    return 'Hello, World!'


def gen(camera):
    while True:
        frame = camera.get_jpeg()
        # time.sleep(1.0)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    #return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(gen2(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed2')
def video_feed2():
    output = gen3(VideoCamera())
    return success_handle(output)

#@app.before_first_request
# def activate_job():
#     def run_job():
#         while True:
#             print("Run recurring task")
#             time.sleep(3)
#
#     thread = threading.Thread(target=run_job)
#     thread.start()

@app.route('/request_count')
def request_count():
    value = cache.get('global_obj_count')
    value_time = cache.get('global_value_time')
    info = {"pers": str(value), "date": value_time}

    def customconverter(o):
        if isinstance(o, datetime.datetime):
            return o.__str__()

    output = json.dumps(info, default = customconverter)
    return success_handle(output)


def gen2(camera):
    while True:
        time.sleep(1.0)
        label = '{:%d.%m.%Y %H:%M:%S}'.format(datetime.datetime.now())
        frame = camera.get_frame()
        if np.shape(frame) == ():
            print('Empty frame: ' + label)
            continue
        frame = imutils.resize(frame, width=400)
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()
        #obj_count = 0

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            # args["confidence"]
            if confidence > confidence_val:
                # extract the index of the class label from the
                # `detections`, then compute the (x, y)-coordinates of
                # the bounding box for the object
                idx = int(detections[0, 0, i, 1])
                if CLASSES[idx] != "person":
                    continue
                #obj_count += 1
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # draw the prediction on the frame
                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

        label = '{:%d.%m.%Y %H:%M:%S}'.format(datetime.datetime.now())
        cv2.putText(frame, label, (5, 240), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
        ret, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')


def gen3(camera):
    frame = camera.get_frame()
    frame = imutils.resize(frame, width=400)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()
    obj_count = 0

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        # args["confidence"]
        if confidence > confidence_val:
            # extract the index of the class label from the
            # `detections`, then compute the (x, y)-coordinates of
            # the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] != "person":
                continue
            obj_count += 1

    info = {"pers": str(obj_count)};
    info["date"] = datetime.datetime.now()

    def customconverter(o):
        if isinstance(o, datetime.datetime):
            return o.__str__()

    # return json.dumps({"api": '1.0'})
    # return json.dumps({"api": info})
    return json.dumps(info, default = customconverter)


def success_handle(output, status=200, mimetype='application/json'):
    return Response(output, status=status, mimetype=mimetype)


@app.route('/api', methods=['GET'])
def homepage():
    output = json.dumps({"api": '1.0'})
    return success_handle(output)


#def app(environ, start_response):
#    """Simplest possible application object"""
#    data = b'Hello, World!\n'
#    status = '200 OK'
#    response_headers = [
#        ('Content-type', 'text/plain'),
#        ('Content-Length', str(len(data)))
#    ]
#    start_response(status, response_headers)
#    return iter([data])


if __name__ == "__main__":
    cache.set('global_obj_count', 0)
    cache.set('global_value_time', 0)
    # lock to control access to variable
    dataLock = threading.Lock()

    def test_job():
        while True:
            print("Run recurring task")
            time.sleep(3)


    def people_counter_job(camera):
        net2 = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        obj_count = 0
        while True:
            time.sleep(5.0)
            label = '{:%d.%m.%Y %H:%M:%S}'.format(datetime.datetime.now())
            # print('begin get_frame: ' + label)
            frame = camera.get_frame()
            if frame is None:
                print('Empty frame: ' + label)
                continue

            frame = imutils.resize(frame, width=400)
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

            # pass the blob through the network and obtain the detections and
            # predictions
            net2.setInput(blob)
            detections = net2.forward()
            obj_count = 0

            # loop over the detections
            for i in np.arange(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with
                # the prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections by ensuring the `confidence` is
                # greater than the minimum confidence
                # args["confidence"]
                if confidence > confidence_val:
                    # extract the index of the class label from the
                    # `detections`, then compute the (x, y)-coordinates of
                    # the bounding box for the object
                    idx = int(detections[0, 0, i, 1])
                    if CLASSES[idx] != "person":
                        continue
                    obj_count = obj_count + 1

            cache.set('global_obj_count', obj_count)
            cache.set('global_value_time', datetime.datetime.now())
            # print('end get_frame: ' + label)

    # thread handler
    # thread = threading.Thread(target=people_counter_job, args=(VideoCamera(),))
    thread = threading.Timer(1.0, people_counter_job, (VideoCamera(),))
    thread.daemon = True
    thread.start()
    
    print("Run recurring task")
    app.run(host='0.0.0.0', debug=False, threaded=False, use_reloader=False)
