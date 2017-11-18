import numpy as np
import cv2

def create_consensus_image(consensus, image_dimensions):
    consensus_image = np.uint8(np.zeros(image_dimensions))
    for (y, x) in consensus:
        consensus_image[y, x] = 255
    return  consensus_image

def cleanup_consensus(consensus, consensus_mask, green_mask):
    consensus = cv2.morphologyEx(consensus, cv2.MORPH_CLOSE, np.ones((3, 3)))
    consensus = cv2.erode(consensus, np.ones((3, 3)), iterations=2)
    consensus = cv2.bitwise_and(consensus, consensus_mask)
    consensus = cv2.bitwise_and(consensus, green_mask)
    return consensus

def draw_detected_road_region(rgb_image, consensus):
    _, contours, _ = cv2.findContours(consensus, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if (len(contours) == 0):
        return None

    biggest_contour = contours[0]
    for c in contours:
        if (cv2.contourArea(c) > cv2.contourArea(biggest_contour)):
            biggest_contour = c

    biggest_contour = cv2.convexHull(biggest_contour)
    epsilon = 0.01 * cv2.arcLength(biggest_contour, True)
    biggest_contour = cv2.approxPolyDP(biggest_contour, epsilon, True)

    return cv2.drawContours(rgb_image, [biggest_contour], -1, (0, 0, 255), 3)