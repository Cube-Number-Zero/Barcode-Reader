import barcode_finder as finder
import cv2


def read_barcode(path_to_image):
    image = finder.find_barcode(cv2.imread(path_to_image), True)
    filtered = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = filtered
    print(image)
    cv2.imshow("Barcode", image)

    
    return NotImplemented

if __name__ == "__main__":
    path_to_image = "barcode can.jpg"
    read_barcode(path_to_image)
