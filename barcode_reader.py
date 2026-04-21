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

    cv2.imshow("", cv2.line(image, (0, start_pos_y), (x, y), 5))
    return return_array
    

def read_barcode(path_to_image):
    # get cropped barcode image from barcode_finder
    image = finder.find_barcode(cv2.imread(path_to_image))
    # convert to grayscale and apply threshold
    filtered = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, filtered = cv2.threshold(filtered, 127, 255, cv2.THRESH_BINARY)
    # get scan data as list
    scan_data = scan_line(filtered)

    # stupid evil code this point forward (only works in one test case)
    # reformat scan_data to lengths of series of 0's or 1's with no indication
    # as to which is which (I will be assuming it always starts on white)
    scan_data_formatted = []
    flag = 0
    count = 0
    for px in scan_data:
        if px == flag:
            count += 1
        else:
            scan_data_formatted.append(count)
            count = 1
            if flag == 0: flag = 1
            else: flag = 0
    # remove white space left of the image (sketchy)
    scan_data_formatted.pop(0)
    # get average bar length from left and right sides of barcode
    average_bar = (scan_data_formatted[0] + scan_data_formatted[1] + scan_data_formatted[2] + \
                  scan_data_formatted[-1] + scan_data_formatted[-2] + scan_data_formatted[-3]) // 6
    # remove left and right sides of barcode
    del scan_data_formatted[:3]
    del scan_data_formatted[-3:]
    # remove middle thing
    del scan_data_formatted[16:21]
    # reformat barcode again to get it even more perfect
    for i, strip in enumerate(scan_data_formatted):
        scan_data_formatted[i] = strip // average_bar
    # read barcode
    patterns = [scan_data_formatted[i:i+4] for i in range(0, len(scan_data_formatted), 4)]
    encoding = {(3,2,1,1): 0, (2,2,2,1): 1, (2,1,2,2): 2, (1,4,1,1): 3, (1,1,3,2):4, (1,2,3,1): 5, (1,1,1,4): 6, (1,3,1,2): 7, (1,2,1,3): 8, (3,1,1,2): 9}
    code = []
    for pattern in patterns:
        code.append((encoding[tuple(pattern)]))
    # do checksum
    checksum = 0
    # Checksum calculations are not consistent across all UPC and EAN codes.
    # Basically, this doesn't work all the time.
    for i in range(1, len(code)-1, 2):
        checksum += code[i]
    checksum = checksum * 3
    for i in range(0, len(code), 2):
        checksum += code[i]
    if checksum % 10 == code[-1]:
        checksum = True
    else:
        checksum = False
    return "".join([str(x) for x in code])

if __name__ == "__main__":
    path_to_image = "barcode perfection2.gif"
    print(read_barcode(path_to_image))
