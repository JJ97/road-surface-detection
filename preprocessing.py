import numpy as np
import cv2

# lower and upper bounds for defining 'green' when masking for vegetation
LOWER_GREEN = np.array([40, 50, 60])
UPPER_GREEN = np.array([80, 255, 255])

def downsample(left, right, rate):
    """ Decrease the size of input images so that they can be processed quicker at the expense of accuracy

    Arguments:
        left  -- image from the left stereo camera
        right -- image from the right stereo camera
        rate  -- downsampling coefficient in range 0->1
                 e.g. 0.5 to half the height and width of both images
    """
    left = cv2.resize(left, (0, 0), fx=rate, fy=rate)
    right = cv2.resize(right, (0, 0), fx=rate, fy=rate)
    return (left, right)

def crop_sky(left, right, sky_crop_height):
    """ Remove the uppermost pixels from each image for faster processing
        These regions will realistically never be road anyway and can be ignored

    Arguments:
        left             -- image from the left stereo camera
        right            -- image from the right stereo camera
        sky_crop_height  -- number of rows to remove from the top of each image
    """
    left_cropped = left[sky_crop_height:,:,:]
    right_cropped = right[sky_crop_height:,:,:]
    return (left_cropped, right_cropped)

def get_green_mask(imgL):
    """ Use HSV to create a mask for removing vegetation
        All pixels within a certain threshold of 'green' are black, otherwise white

    Arguments:
        imgL -- image from the left stereo camera
    """
    # convert to HSV color space
    hsv = cv2.cvtColor(imgL, cv2.COLOR_BGR2HSV)
    # threshold for values between LOWER_GREEN and UPPER_GREEN
    thresh = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    # dilate the resulting image to account for noise
    thresh = cv2.dilate(thresh, np.ones((3, 3)), iterations=3)
    # flip image bits so that it can be ANDed with disparity to keep all non-green pixels
    mask = cv2.bitwise_not(thresh)
    return mask


def create_edge_mask(imgL, canny_mask):
    """ Create a mask for separating road regions from pavement and other objects
        i.e. separating consensus points if a strong edge is between them

    Arguments:
        imgL       -- image from the left stereo camera
        canny_mask -- used to ignore center of image so that road markings do not become an issue
    """
    canny = cv2.Canny(imgL, 100, 300, L2gradient=True)
    # dilate so that edges are more likely to connect, completely cutting two consensus regions in two
    canny = cv2.dilate(canny, np.ones((3, 3)))
    # mask out center of canny so that road markings do not also split up consensus regions
    edge_mask = cv2.bitwise_and(canny, cv2.bitwise_not(canny_mask))
    # flip image bits so that it can be ANDed with consensus image to keep all pixels not on an edge
    edge_mask = cv2.bitwise_not(edge_mask)
    return edge_mask


def cleanup_disparity(disparity, disparity_mask, green_mask):
    """ Improve disparity results before further processing i.e. filtering noise, infilling and masking ROI

    Arguments:
        disparity       -- disparity obtained from left and right images
        disparity_mask  -- used to ignore regions that will never be road e.g. sky, car bumper
        green_mask      -- used to ignore vegetation
    """
    # filter out spuriously high disparities due to motion blur
    disparity[disparity >= 50] = 0

    # get mask of all points in disparity image where no actual disparity was found
    _, holes_in_disparity = cv2.threshold(disparity, 1, 1, cv2.THRESH_BINARY_INV)
    holes_in_disparity = np.uint8(holes_in_disparity)
    # fill in these holes for a more consistent image
    disparity_clean = cv2.inpaint(disparity, holes_in_disparity, 5, cv2.INPAINT_NS)

    # apply masks to remove disparities that are very likely to not be road
    # e.g. sky, car bumper, grass
    disparity_clean = cv2.bitwise_and(disparity_clean, disparity_mask)
    disparity_clean = cv2.bitwise_and(disparity_clean, green_mask)
    return disparity_clean, holes_in_disparity
