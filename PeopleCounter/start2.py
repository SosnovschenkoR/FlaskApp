import argparse
import datetime
import json
import time
import threading

import imutils as imutils
import numpy as np
import cv2

from flask import Flask, render_template, Response
from cachelib.simple import SimpleCache
from camera2 import VideoCamera

app = Flask(__name__)

cache = SimpleCache()

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


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
    # return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(gen2(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed2')
def video_feed2():
    output = gen3(VideoCamera())
    return success_handle(output)


# @app.before_first_request
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
    info = {"pers": str(value), "date": datetime.datetime.now()}

    def customconverter(o):
        if isinstance(o, datetime.datetime):
            return o.__str__()

    output = json.dumps(info, default=customconverter)
    return success_handle(output)


def gen2(camera):
    while True:
        frame = camera.get_frame()
        if np.shape(frame) == ():
            continue
        time.sleep(1.0)
        frame = imutils.resize(frame, width=400)
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()
        # obj_count = 0

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > args["confidence"]:
                # extract the index of the class label from the
                # `detections`, then compute the (x, y)-coordinates of
                # the bounding box for the object
                idx = int(detections[0, 0, i, 1])
                if CLASSES[idx] != "person":
                    continue
                # obj_count += 1
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # draw the prediction on the frame
                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

        label = '{:%d.%m.%Y %H:%M:%S}'.format(datetime.datetime.now())
        cv2.putText(frame, label, (0, 290), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)

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
        if confidence > args["confidence"]:
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
    return json.dumps(info, default=customconverter)


def success_handle(output, status=200, mimetype='application/json'):
    return Response(output, status=status, mimetype=mimetype)


@app.route('/api', methods=['GET'])
def homepage():
    output = json.dumps({"api": '1.0'})
    return success_handle(output)


if __name__ == "__main__":
    cache.set('global_obj_count', 0)


    def test_job():
        while True:
            print("Run recurring task")
            time.sleep(3)


    def people_counter_job(camera):
        while True:
            frame = camera.get_frame()
            time.sleep(1.0)
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
                if confidence > args["confidence"]:
                    # extract the index of the class label from the
                    # `detections`, then compute the (x, y)-coordinates of
                    # the bounding box for the object
                    idx = int(detections[0, 0, i, 1])
                    if CLASSES[idx] != "person":
                        continue
                    obj_count = obj_count + 1
            cache.set('global_obj_count', obj_count)


    # thread = threading.Thread(target=people_counter_job, args=(VideoCamera(),))
    # thread.daemon = True
    # thread.start()

    app.run(host='127.0.0.1', debug=True, threaded=True, use_reloader=False)
