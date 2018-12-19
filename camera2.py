import cv2
import numpy


class VideoCamera(object):
    class_var = 0
    video = 0
    url = 'http://127.0.0.1:8081/video.mjpg'
    count = 0

    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        #if VideoCamera.class_var == 0:
            #VideoCamera.video = cv2.VideoCapture(0)
            #VideoCamera.video = cv2.VideoCapture(self.url)
        self.video = cv2.VideoCapture(self.url)
        #self.video = cv2.VideoCapture(0)

        VideoCamera.class_var += 1
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')

    def __del__(self):
        VideoCamera.class_var -= 1
        #if VideoCamera.class_var == 0:
        #    VideoCamera.video.release()
        self.video.release()


    def get_frame(self):
        #success, frame = VideoCamera.video.read()
        success, frame = self.video.read()
        #cv2.imwrite("frame%d.jpg" % self.count, image)
        #self.count += 1
        return frame


    def get_jpeg(self):
        #success, image = VideoCamera.video.read()
        success, image = self.video.read()

        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
