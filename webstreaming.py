
import threading
import argparse
import datetime
import imutils
import time
import cv2
from flask import Flask
from flask import render_template
from flask import Response
from imutils.video import VideoStream

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful when multiple browsers/tabs
# are viewing the stream)
outputFrame = None
lock = threading.Lock()

# initialize a Flask object
app = Flask(__name__)

# initialize the video stream and allow the camera sensor to warmup
vs = VideoStream(src=0).start()
time.sleep(2.0)


@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")
