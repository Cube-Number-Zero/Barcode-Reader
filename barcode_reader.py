import barcode_finder as finder
import cv2

image = finder.find_barcode(cv2.imread("barcode cow.jpg"))
cv2.imshow("Barcode", image)
