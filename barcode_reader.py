import barcode_finder as finder
import cv2


def read_barcode(path_to_image):
    image = finder.find_barcode(cv2.imread(path_to_image), True)
    filtered = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = filtered

    scan_depth = 50

    x, y = image.shape

    total_len = x + y

    scan_array = []

    for i in range(scan_depth):
        tmp_list = []
        scan_point = [0, 0]
        #starting point
        if int(i / scan_depth * total_len) < y - 1:
            scan_point = [0, int(i / scan_depth * total_len)]
        else:
            scan_point = [int(i / scan_depth * total_len) - y, y - 1]
            for j in range(int(i / scan_depth * total_len) - y - 1):
                tmp_list.append(-1)
        #Log and move
        while True:
            if image[scan_point[0], scan_point[1]] > 128:
                tmp_list.append(1)
            else:
                tmp_list.append(0)
            #tmp_list.append(image[scan_point[0], scan_point[1]])
            scan_point[0] += 1
            scan_point[1] -= 1
            #Edge Checks
            if scan_point[0] >= x or scan_point[1] < 0:
                break
        scan_array.append(tmp_list)

    #print(image)
    with open("tmp.txt", "w") as f:
        print(scan_array, file = f)
    cv2.imshow("Barcode", image)

    
    return NotImplemented

if __name__ == "__main__":
    path_to_image = "barcode can.jpg"
    read_barcode(path_to_image)
