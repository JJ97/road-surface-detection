import cv2
import os
import numpy as np

#####################################################################

# Example : load, display and compute SGBM disparity
# for a set of rectified stereo images from a  directory structure
# of left-images / right-images with filesname DATE_TIME_STAMP_{L|R}.png

# basic illustrative python script for use with provided stereo datasets

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2017 Department of Computer Science,
#                    Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

#####################################################################

# setup the disparity stereo processor to find a maximum of 128 disparity values
# (adjust parameters if needed - this will effect speed to processing)

# uses a modified H. Hirschmuller algorithm [Hirschmuller, 2008] that differs (see opencv manual)
# parameters can be adjusted, current ones from [Hamilton / Breckon et al. 2013]

# FROM manual: stereoProcessor = cv2.StereoSGBM(numDisparities=128, SADWindowSize=21);

# From help(cv2): StereoBM_create(...)
#        StereoBM_create([, numDisparities[, blockSize]]) -> retval
#
#    StereoSGBM_create(...)
#        StereoSGBM_create(minDisparity, numDisparities, blockSize[, P1[, P2[,
# disp12MaxDiff[, preFilterCap[, uniquenessRatio[, speckleWindowSize[, speckleRange[, mode]]]]]]]]) -> retval

MAX_DISPARITY = 128
STEREO_PROCESSOR = cv2.StereoSGBM_create(minDisparity=0, numDisparities=MAX_DISPARITY, blockSize=21)

def calculate_disparity(left_image, right_image):
    grayL = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

    # compute disparity image from undistorted and rectified stereo images
    # that we have loaded
    # (which for reasons best known to the OpenCV developers is returned scaled by 16)

    disparity = STEREO_PROCESSOR.compute(grayL, grayR)


    # filter out noise and speckles (adjust parameters as needed)

    dispNoiseFilter = 5  # increase for more aggressive filtering
    cv2.filterSpeckles(disparity, 0, 4000, MAX_DISPARITY - dispNoiseFilter)

    # scale the disparity to 8-bit for viewing
    # divide by 16 and convert to 8-bit image (then range of values should
    # be 0 -> max_disparity) but in fact is (-1 -> max_disparity - 1)
    # so we fix this also using a initial threshold between 0 and max_disparity
    # as disparity=-1 means no disparity available
    #
    _, disparity = cv2.threshold(disparity, 0, MAX_DISPARITY * 16, cv2.THRESH_TOZERO)
    disparity = (disparity / 16.).astype(np.uint8)

    return disparity

