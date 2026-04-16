import barcode_finder as finder
import numpy as np
import cv2


def scan_line(image, start_pos_y, angle=0):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    t = np.tan(np.radians(-angle))
    winname = "Barcode"
    y = start_pos_y
    x = 0

    return_array = []
    
    while x < image.shape[1]:
        return_array.append(image[y, x] // 255)
        x += 1
        y = round(start_pos_y + x * t)

    return return_array
    

def read_barcode(path_to_image):
    image = finder.find_barcode(cv2.imread(path_to_image))
    filtered = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = filtered
    print(image)
    cv2.imshow("Barcode", image)

    
    return NotImplemented

if __name__ == "__main__":
    path_to_image = "barcode can.jpg"
    read_barcode(path_to_image)
