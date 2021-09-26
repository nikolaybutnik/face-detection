
from flask import Flask, render_template, Response
import cv2


app = Flask(__name__)

webcam = cv2.VideoCapture(0)


def generate_frames():
    while True:
        # read retruns two params. 1: whether it successfully returns a frame (bool) 2: actual frame
        success, frame = webcam.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")


@app.route("/video_stream")
def video_stream():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
