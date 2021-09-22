import cv2

# Load some pre-trained data on face frontals from opencv
#  The algorithm prioritizes speed over accuracy. Ensure photos have good lighting.
trained_face_data = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Choose image to detect faces
img = cv2.imread('people.jpeg')

# Convert image to grayscale
grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces in grayscale image by running image through the algo rithm
# Return coordinates of the rectangles enclosing objects (faces in this case)
# Example: [[135 127 190 190]] - Top left corner is at 135 and 127. Rectangle
# of size 190x190 would be drawn between these coordinates.
face_coordinates = trained_face_data.detectMultiScale(grayscale_img)

# Draw rectangles around faces. Note: opencv uses BGR instead of RGB
# Example: cv2.rectangle(image, top left coordinate = (x,y), bottom right
# coordinate = (x + width, y + height), line color, line thickness)
# cv2.rectangle(img, (135, 127), (135+190, 135+190), (0, 255, 0), 2)
# Loop through every detected face in array
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Open image in a window
cv2.imshow('Python Face Detector', img)

# Halt code execution until a key is pressed, to avoid imediately
# terminating the program and closing the image
cv2.waitKey()

print('If this prints, the code succeeded')
