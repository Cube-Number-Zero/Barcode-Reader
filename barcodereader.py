import cv2
import numpy as np
img = cv2.imread("barcode test.jpg", cv2.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read"

threshold, filtered_img = cv2.threshold(cv2.GaussianBlur(img,(5,5),0),0,255,\
                                       cv2.THRESH_BINARY+cv2.THRESH_OTSU)

print(filtered_img.shape)
cv2.imshow("Display window", cv2.resize(filtered_img, (filtered_img.shape[1]//2, filtered_img.shape[0]//2)))
