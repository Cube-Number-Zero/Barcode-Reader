import cv2
import numpy as np
import math

def find_barcode(img, rotated=False):
    assert img is not None, "file could not be found"

    # Resize image for consistency
    image_y, image_x, _ = img.shape
    scale = max(math.ceil(image_x / 1000), math.ceil(image_y / 1000))
    image = cv2.resize(img, (image_x // scale, image_y // scale))

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
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2))
        filtered = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)

        # Merge nearby areas into one shape
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 18))
        filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel)

        # Reduce small details
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 10))
        filtered = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    else:
        # 
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 10))
        filtered = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)

        # Merge nearby areas into one shape
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 8))
        filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel)

        # Reduce small details
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 30))
        filtered = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)

    # Find contours of shape (not finished)
    contours, _ = cv2.findContours(filtered, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    try:
        best_contour = max(contours, key=cv2.contourArea)
    except ValueError:
        raise ValueError("Barcode could not be found. Try rotating the image?")

    # Display image
    x, y, w, h = cv2.boundingRect(best_contour)
    tolerance_x = 5
    tolerance_y = 5
    # ensuring tolerance doesnt make it try to go out of the bounds
    # of the image, assigning each axis test case to a variable
    oob_x = (x-tolerance_x < 0) or (x+w+tolerance_x > len(image[1]))
    oob_y = (y-tolerance_y < 0) or (y+h+tolerance_y > len(image[0]))
    if oob_x or oob_y:
        # these individual cases allow one tolerance to still work
        # if another one cannot
        if oob_x and not oob_y:
            image = image[y-tolerance_y:y+h+tolerance_y, x:x+w]
        elif oob_y and not oob_x:
            image = image[y:y+h, x-tolerance_x:x+w+tolerance_x]
        else:
            image = image[y:y+h, x:x+w]
    else:
        image = image[y-tolerance_y:y+h+tolerance_y, x-tolerance_x:x+w+tolerance_x]
    if rotated:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    else:
        return image


if __name__ == "__main__":
    image = cv2.imread("barcode can.jpg")
    image = find_barcode(image, True)
    winname = "Barcode"
    cv2.namedWindow(winname)
    cv2.moveWindow(winname, 40,30)
    cv2.imshow(winname, image)
