import numpy as np
import cv2

HSV_GREEN = 60
LOWER_GREEN = HSV_GREEN - 20
UPPER_GREEN = HSV_GREEN + 20

def get_green_mask(imgL):
    hsv = cv2.cvtColor(imgL, cv2.COLOR_BGR2HSV)
    thresh = cv2.inRange(hsv, np.array([LOWER_GREEN, 75, 30]), np.array([UPPER_GREEN, 255, 255]))
    thresh = cv2.dilate(thresh, np.ones((3, 3)), iterations=3)

    # thresh2 = cv2.inRange(s, 200, 255)
    # thresh2 = cv2.dilate(thresh2, np.ones((3, 3)), iterations=3)
    # #
    # mask = cv2.bitwise_and(thresh,thresh2)
    mask = cv2.bitwise_not(thresh)
    return mask

def convert_illumination_invariant(img, alpha):
    height = img.shape[0]
    width = img.shape[1]

    invariant_image = np.zeros((height,width))

    # transform to 0->1 float from PAQs
    img_float = img.astype('float') / 255.0
    for y in range(height):
        for x in range(width):
            r_component = (1-alpha) * np.log(img_float[y,x,2])
            g_component = 0.5 + np.log(img_float[y,x,1])
            b_component = alpha * np.log(img_float[y, x, 0])
            invariant_image[y,x] = g_component - b_component - r_component

    # transform back to 0->255 int from PAQs
    return (invariant_image * 255).astype('uint8')


def mask_canny_for_post_processing(img, canny_mask):
    canny = cv2.Canny(img, 100, 300, L2gradient=True)
    canny = cv2.dilate(canny, np.ones((3, 3)))
    canny = cv2.bitwise_and(canny, cv2.bitwise_not(canny_mask))
    canny = cv2.bitwise_not(canny)
    return canny


def cleanup_disparity(disparity, disparity_mask):
    # disparity[disparity >= 50] = 0
    _, holes_in_disparity = cv2.threshold(disparity, 1, 1, cv2.THRESH_BINARY_INV)
    holes_in_disparity = np.uint8(holes_in_disparity)

    disparity_clean = cv2.inpaint(disparity, holes_in_disparity, 5, cv2.INPAINT_NS)

    disparity_clean = cv2.bitwise_and(disparity_clean, disparity_mask)
    return disparity_clean, holes_in_disparity
