import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize hand detector
detector = HandDetector(maxHands=1)

# Initialize classifier with model and labels
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20  # Margin around the hand
imgSize = 300  # Size of the output image

folder = "Data/C"  # Directory to save images (not used here)
counter = 0  # Counter for saved images (not used here)

labels = ["A", "B", "C"]  # Labels for the classifier

while True:
    success, img = cap.read()  # Capture frame from webcam
    imgOutput = img.copy()  # Make a copy of the frame for output
    hands, img = detector.findHands(img)  # Detect hands in the frame
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']  # Get bounding box coordinates of the hand

        # Create a white image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        
        # Crop the hand image with offset
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            # Adjust width to maintain aspect ratio
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        else:
            # Adjust height to maintain aspect ratio
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        # Draw rectangle around the hand and display the predicted label
        cv2.rectangle(imgOutput, (x - offset, y - offset-50),
                      (x - offset+90, y - offset-50+50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x-offset, y-offset),
                      (x + w+offset, y + h+offset), (255, 0, 255), 4)

        # Display images
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)  # Show the output frame
    cv2.waitKey(1)
