
from flask import Flask, render_template, Response
import cv2
from flask.wrappers import Request


# Load some pre-trained data on face frontals from opencv
# The algorithm prioritizes speed over accuracy. Ensure photos have good lighting.
trained_face_data = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
trained_eye_data = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml')


app = Flask(__name__)

# Capture webcam footage. Passing in 0 targets default webcam.
# It's possible to pass in a string with the target video name instead.
webcam = cv2.VideoCapture(0)


def generate_frames():
    while True:
        # Read returns two params. 1: whether it successfully returns a frame (bool) 2: actual frame
        success, frame = webcam.read()

        # Convert captured frame to grayscale
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces. This function can also take an argument to adjust sensitivity.
        face_coordinates = trained_face_data.detectMultiScale(
            grayscale_frame, minNeighbors=20, minSize=[100, 100])
        eye_coordinates = trained_eye_data.detectMultiScale(
            grayscale_frame, minNeighbors=40)

        # Draw rectangles around faces
        for (x, y, w, h) in face_coordinates:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        for (x, y, w, h) in eye_coordinates:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/stream_start')
def stream_start():
    print("The stream has started")
    return("nothing")


@app.route('/stream_stop')
def stream_stop():
    webcam.release()
    return render_template("index.html", stream=False)


@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html", is_streaming=True)


@app.route("/video_stream")
def video_stream():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
