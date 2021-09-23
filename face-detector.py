import cv2

# Load some pre-trained data on face frontals from opencv
# The algorithm prioritizes speed over accuracy. Ensure photos have good lighting.
trained_face_data = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# FOR STILL IMAGES

# To analyze image - choose image to detect faces
# img = cv2.imread('people.jpeg')

# Convert image to grayscale
# grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces in grayscale image by running image through the algo rithm
# Return coordinates of the rectangles enclosing objects (faces in this case)
# Example: [[135 127 190 190]] - Top left corner is at 135 and 127. Rectangle
# of size 190x190 would be drawn between these coordinates.
# face_coordinates = trained_face_data.detectMultiScale(grayscale_img)

# Draw rectangles around faces. Note: opencv uses BGR instead of RGB
# Example: cv2.rectangle(image, top left coordinate = (x,y), bottom right
# coordinate = (x + width, y + height), line color, line thickness)
# cv2.rectangle(img, (135, 127), (135+190, 135+190), (0, 255, 0), 2)
# Loop through every detected face in array
# for (x, y, w, h) in face_coordinates:
#     cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Open image in a window
# cv2.imshow('Python Face Detector', img)

# Halt code execution until a key is pressed, to avoid imediately
# terminating the program and closing the image
# cv2.waitKey()

# print('If this prints, the code succeeded')

# FOR VIDEO

# Capture webcam footage. Passing in 0 targets default webcam.
# It's possible to pass in a string with the target video name instead.
webcam = cv2.VideoCapture(0)

# To analyze the webcame footage, we need to iterate infinitely over the
# video frames.
while True:
    # Read the current frame. First value is a boolean.
    successful_frame_read, frame = webcam.read()

    # Convert captured frame to grayscale
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscale_frame)

    # Draw rectangles around faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Python Face Detector', frame)

    # If nothing is passed into waitKey, it'll wait for input infinitely.
    # By passing number 1, the program will wait 1 millisecond before automatically
    # hitting a key and advancing. Each frame will be displayed for 1 millisecond.
    key = cv2.waitKey(1)

    # If Q key is pressed, exit the program (these are ascii codes for q and Q)
    if key == 81 or key == 113:
        break
# Release VideoCapture object
webcam.release()


print('If this prints, the code succeeded')
