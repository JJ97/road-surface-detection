#####################################################################

# Example : project SGBM disparity to 3D points for am example pair
# of rectified stereo images from a  directory structure
# of left-images / right-images with filesname DATE_TIME_STAMP_{L|R}.png

# basic illustrative python script for use with provided stereo datasets

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2017 Deparment of Computer Science,
#                    Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

#####################################################################

import cv2
import os
import numpy as np
import random
import csv

##########################################

# fixed camera parameters for this stereo setup (from calibration)

camera_focal_length_px = 399.9745178222656  # focal length in pixels
camera_focal_length_m = 4.8 / 1000          # focal length in metres (4.8 mm)
stereo_camera_baseline_m = 0.2090607502     # camera baseline in metres

image_centre_h = 262.0
image_centre_w = 474.5


#####################################################################

## project_disparity_to_3d : project a given disparity image
## (uncropped, unscaled) to a set of 3D points with optional colour

#####################################################################
def project_disparity_to_3d(disparity, max_disparity):
    height, width = disparity.shape[:2]

    points = []
    for y, row in enumerate(disparity): # 0 - height is the y axis index
        for x, p in enumerate(row): # 0 - width is the x axis index

            # if we have a valid non-zero disparity
            if (p > 0):
                # calculate corresponding 3D point [X, Y, Z]

                # stereo lecture - slide 22 + 25
                Z = ((camera_focal_length_px * stereo_camera_baseline_m) / p)
                X = ((x - image_centre_w) * Z) / camera_focal_length_px
                Y = ((y - image_centre_h) * Z) / camera_focal_length_px

                # add to points

                points.append([X,Y,Z])

    return np.array(points)

# project a set of 3D points back the 2D image domain
# sacrifice readability for a small increase in efficiency
def project_3D_points_to_2D_image_points(points):
    return [ [((p[1] * camera_focal_length_px) / p[2]) + image_centre_h,
              ((p[0] * camera_focal_length_px) / p[2]) + image_centre_w] for p in points ]

