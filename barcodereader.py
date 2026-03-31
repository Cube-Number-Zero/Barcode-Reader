import cv2
import numpy as np
img = cv2.imread("barcode can.jpg")
assert img is not None, "file could not be read"





# convert to grayscale
# calculate partial derivatives in x- and y-direction (sobel)
# get absolute different from x and y gradients
# blur
# threshold
# dilate (30, 10)
# erode (30, 10)
# erode then dilate to remove noise
# find contours using forbidden magicks

image = cv2.resize(img, None, fx=0.7, fy=0.7, interpolation = cv2.INTER_CUBIC)


cv2.imshow("Display window", cv2.resize(image, (image.shape[1]//4, image.shape[0]//4)))
