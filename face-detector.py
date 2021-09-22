import cv2

# Load some pre-trained data on face frontals from opencv
trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

# Choose image to detect faces
img = cv2.imread('Elon_Musk.jpeg')
# Convert image to grayscale
grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Open image in a window
cv2.imshow('Python Face Detector', grayscale_img)
# Halt code execution until a key is pressed, to avoid imediately
# terminating the program and closing the image
cv2.waitKey()

print('If this prints, the code succeeded')
