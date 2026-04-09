import cv2
import numpy as np
import math

def find_barcode_directional(img, rotated=False):
    assert img is not None, "file could not be found"

    # Resize image for consistency
    original_y, original_x, _ = img.shape
    scale = max(math.ceil(original_x / 1000), math.ceil(original_y / 1000))
    image = cv2.resize(img, (original_x // scale, original_y // scale))

    # Convert to grayscale
    filtered = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur to reduce noise
    filtered = cv2.GaussianBlur(filtered, (3,3), 0)

    # Find the gradient using sobel derivatives
    partial_derivative_x = cv2.convertScaleAbs(cv2.Sobel(filtered, cv2.CV_16S, 1, 0))
    partial_derivative_y = cv2.convertScaleAbs(cv2.Sobel(filtered, cv2.CV_16S, 0, 1))

    # Find areas with large derivates in X but not Y to locate vertical edges
    if rotated:
        filtered = cv2.subtract(partial_derivative_y, partial_derivative_x)
    else:
        filtered = cv2.subtract(partial_derivative_x, partial_derivative_y)

    # Otsu binarization (Threshold)
    _, filtered = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if rotated:
        # Merge nearby areas into one shape
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 18))
        filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel)

        # Reduce small details
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 10))
        filtered = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    else:
        # Merge nearby areas into one shape
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 8))
        filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel)

        # Reduce small details
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 30))
        filtered = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)

    # Find contours of shape (not finished)
    filtered = cv2.resize(filtered, (original_x, original_y))
    contours, _ = cv2.findContours(filtered, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    best_contour = max(contours, key=cv2.contourArea)

    # Display image
    x, y, w, h = cv2.boundingRect(best_contour)
    image = image[y:y+h, x:x+w]
    return image


if __name__ == "__main__":
    image = cv2.imread("barcode cow.jpg")
    image = find_barcode_directional(image, True)
    winname = "Barcode"
    cv2.namedWindow(winname)
    cv2.moveWindow(winname, 40,30)
    cv2.imshow(winname, image)
