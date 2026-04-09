import cv2
import numpy as np
import math

# fetch image
img = cv2.imread("barcode test.jpg")
assert img is not None, "file could not be found"

# Resize image for consistency
y, x, _ = img.shape
scale = max(math.ceil(x / 1000), math.ceil(y / 1000))
image = cv2.resize(img, (x // scale, y // scale))

# Convert to grayscale
filtered = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Blur to reduce noise
filtered = cv2.GaussianBlur(filtered, (3,3), 0)

# Find the gradient using sobel derivatives
partial_derivative_x = cv2.convertScaleAbs(cv2.Sobel(filtered, cv2.CV_16S, 1, 0))
partial_derivative_y = cv2.convertScaleAbs(cv2.Sobel(filtered, cv2.CV_16S, 0, 1))

# Find areas with large derivates to locate edges
filtered = cv2.absdiff(partial_derivative_x, partial_derivative_y)

# Otsu binarization (Threshold)
_, filtered = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Merge nearby areas into one shape
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel)

# Reduce small details
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
filtered = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)

# Find contours of shape (not finished)
contours, _ = cv2.findContours(filtered, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
best_contour = max(contours, key=cv2.contourArea)

# Display image
x, y, w, h = cv2.boundingRect(best_contour)
#image = image[y:y+h, x:x+w]
#image = cv2.drawContours(image, best_contour, -1, (255, 0, 255), 8)
image = filtered
winname = "Barcode"
cv2.namedWindow(winname)
cv2.moveWindow(winname, 40,30)
cv2.imshow(winname, image)
