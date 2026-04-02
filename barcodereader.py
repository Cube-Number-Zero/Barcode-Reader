import cv2
import numpy as np
import math
img = cv2.imread("barcode test.jpg")
assert img is not None, "file could not be read"


image = cv2.resize(img, None, fx=0.7, fy=0.7, interpolation = cv2.INTER_CUBIC)

# Convert to grayscale
filtered = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Blur to reduce noise
filtered = cv2.GaussianBlur(filtered, (3,3), 0)

# Find the gradient using sobel derivatives
partial_derivative_x = cv2.convertScaleAbs(cv2.Sobel(filtered, cv2.CV_16S, 1, 0))
partial_derivative_y = cv2.convertScaleAbs(cv2.Sobel(filtered, cv2.CV_16S, 0, 1))

# Find areas with large derivates to locate edges

# test only using x derivative?
#filtered = partial_derivative_x
filtered = cv2.addWeighted(partial_derivative_x, 1.0, partial_derivative_y, 1.0, 0)


_, filtered = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Merge nearby areas into one shape
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel)
# Reduce small details
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
filtered = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)


# Find contours of shape (not finished)
contours, _ = cv2.findContours(filtered, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

image = cv2.drawContours(image, contours, cv2.FILLED, (255, 0, 255), 50)

#image = filtered
y, x, _ = image.shape
scale = max(math.ceil(x / 960), math.ceil(y / 540))
cv2.imshow("Display window", cv2.resize(image, (image.shape[1] // scale, image.shape[0] // scale)))
