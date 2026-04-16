import barcode_finder as finder
import numpy as np
import cv2

def scan_line(image, start_pos_y=None, angle=0):
    """
    param image: a binarized cv2 image object
    param start_pos_y: the y pos to start the scan from. Default is none, which
                       starts it halfway up the image.
    param angle: the angle in degrees the scan should be at, default 0 is level
    returns: list of ints 0 or 1 corresponding to pixel values from the scanned
             line. 0 for white, 1 for black
    """
    if start_pos_y is None:
        start_pos_y = image.shape[0] // 2
    t = np.tan(np.radians(-angle))
    y = start_pos_y
    x = 0

    return_array = []
    
    while x < image.shape[1]:
        return_array.append(1-int(image[y, x] // 255))
        x += 1
        y = round(start_pos_y + x * t)

    #cv2.imshow("", cv2.line(image, (0, start_pos_y), (x, y), 5))
    return return_array
    

def read_barcode(path_to_image):
    # get cropped barcode image from barcode_finder
    image = finder.find_barcode(cv2.imread(path_to_image))
    filtered = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, filtered = cv2.threshold(filtered, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow("Barcode", filtered)
    print(scan_line(filtered))
    return NotImplemented

if __name__ == "__main__":
    path_to_image = "barcode screenshot.png"
    read_barcode(path_to_image)
