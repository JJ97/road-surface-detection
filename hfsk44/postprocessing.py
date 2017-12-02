import numpy as np
import cv2

def create_consensus_image(consensus, image_dimensions):
    """ Convert consensus points into an image for further processing

    Arguments:
        consensus        -- list of 2D consensus points
        image_dimensions -- size and shape of image to be created
    """
    # create blank image
    consensus_image = np.uint8(np.zeros(image_dimensions))
    # set all pixels with corresponding consensus coordinate to white
    for (y, x) in consensus:
        consensus_image[int(y), int(x)] = 255
    return consensus_image

def cleanup_consensus(consensus, edge_mask):
    """ Refine consensus using mask derived from canny and morphology functions

    Arguments:
        consensus -- image of consensus points
        edge_mask -- mask used to split consensus image along strong edges in original image
                     e.g. side of road, car wheel
    """
    # morphology functions applied to fill gaps / banding effects in consensus due to noise
    consensus = cv2.morphologyEx(consensus, cv2.MORPH_CLOSE, np.ones((3, 3)))
    consensus = cv2.erode(consensus, np.ones((3, 3)), iterations=2)
    # apply mask to remove consensus points that lie on a strong edge in original image
    consensus = cv2.bitwise_and(consensus, edge_mask)
    return consensus

def draw_detected_road_region(imgL, consensus):
    """ Draw convex hull of largest consensus region onto left image for display
        Also find center point of road region

    Arguments:
        imgL      -- image from left side camera
        consensus -- image of consensus points
    """

    # find largest continuous consensus region
    _, contours, _ = cv2.findContours(consensus, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if (len(contours) == 0):
        return None
    biggest_contour = max(contours, key=cv2.contourArea)

    # take its convex hull and approximate it to a polygon for neater display
    biggest_contour = cv2.convexHull(biggest_contour)
    epsilon = 0.01 * cv2.arcLength(biggest_contour, True)
    biggest_contour = cv2.approxPolyDP(biggest_contour, epsilon, True)

    # draw the road contour onto the image
    imgL = cv2.drawContours(imgL, [biggest_contour], -1, (0, 0, 255), 3)

    # calculate the center point of the road region
    M = cv2.moments(biggest_contour)
    centroid = np.array([int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])])

    return imgL, centroid

def draw_normal_vector(imgL, normal, centroid):
    """ Draw normal vector adjusted

    Arguments:
        imgL      -- image from left side camera
        normal    -- two sets of points representing the normal line projected from 3D
        centroid  -- center of the road region
    """
    # convert normal to correct form for drawing
    a = np.array(list(map(int, reversed(normal[0]))))
    b = np.array(list(map(int, reversed(normal[1]))))

    # adjust normal vector to center of road region
    b += centroid - a

    return cv2.line(imgL, tuple(centroid), tuple(b), (0, 255, 0), 5)


def uncrop(res, imgL, sky_crop_height):
    """ Add the sky region back into result image

    Arguments:
        res             -- result image with road region drawn on
        imgL            -- image from left side camera
        sky_crop_height -- number of rows to add back from the top
    """
    # cut out the sky
    sky = imgL[:sky_crop_height,:,:]
    # stack it on top of the result
    res = np.vstack((sky,res))
    return res

