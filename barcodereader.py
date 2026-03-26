import cv2 as cv
import numpy as np
img = cv.imread("barcode test.jpg", cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read"

threshold, filtered_img = cv.threshold(cv.GaussianBlur(img,(5,5),0),0,255,\
                                       cv.THRESH_BINARY+cv.THRESH_OTSU)

print(filtered_img.shape)
cv.imshow("Display window", cv.resize(filtered_img, (filtered_img.shape[1]//2, filtered_img.shape[0]//2)))
