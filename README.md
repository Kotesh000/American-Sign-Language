Hand Sign Detection language

Libraries and Modules
cv2: OpenCV, a library for computer vision tasks.
cvzone.HandTrackingModule: Part of the cvzone library for hand detection.
numpy (np): A library for numerical operations.
math: Standard Python library for mathematical operations.
time: Standard Python library to handle time-related tasks.

Initialization
cap: Captures video from the default webcam.
detector: Initializes the hand detector to detect a maximum of one hand.
offset: A margin added around the cropped hand image.
imgSize: The size of the final image (300x300 pixels).
folder: Directory where images will be saved.
counter: A counter to keep track of the number of images saved.
