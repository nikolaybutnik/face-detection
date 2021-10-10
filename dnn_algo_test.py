import cv2
import numpy as np
import urllib.request
from matplotlib import pyplot as plt


def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(1, 1, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


# Load pre-trained model. Choose between Caffe and Tensorflow model.
DNN = "CAFFE"
if DNN == "CAFFE":
    modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    configFile = "deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
else:
    modelFile = "opencv_face_detector_uint8.pb"
    configFile = "opencv_face_detector.pbtxt"
    net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

# Load image locally
# image = cv2.imread("Elon_Musk.jpeg")
# Or get image online
url = 'https://static01.nyt.com/images/2019/10/02/video/02-still-for-america-room-loop/02-still-for-america-room-loop-superJumbo.jpg'
req = urllib.request.urlopen(url)
arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
image = cv2.imdecode(arr, -1)


# Get dimensions of input image
(h, w) = image.shape[:2]

# Create 4 dimensional blob from image
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [
                             104., 117., 123.], False, False)

# Set the blob as input and obtain detections
net.setInput(blob)
detections = net.forward()

# Initialize counter for detected faces
detected_faces = 0

# Iterate over all detected faces
for i in range(0, detections.shape[2]):
    # Get confidence/probability of current detections
    confidence = detections[0, 0, i, 2]
    # Only consider detections if greater than certain level of confidence
    if confidence > 0.7:
        detected_faces += 1
        # Get coordinates of current detection
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype('int')
        # Draw detection and confidence
        text = "{:.3f}%".format(confidence*100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY), (255, 0, 0), 3)
        cv2.putText(image, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

# Create dimensions of figure and set title
fig = plt.figure(figsize=(10, 5))
plt.suptitle('Python DNN Face Detection',
             fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')
# Plot images
show_img_with_matplotlib(
    image, 'Faces detected:' + str(detected_faces), 1)
# Show figure
plt.show()
