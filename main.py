import cv2
import os
import csv
import numpy as np
import random
import stereo_disparity
import stereo_to_3d
import ransac
import planar_fitting

DEBUG_WINDOW_SIZE = (300,300)

# where is the data ? - set this to where you have it

MASTER_PATH_TO_DATASET = "/volumes/RAM Disk2/dataset/"; # ** need to edit this **
DIRECTORY_TO_CYCLE_LEFT = "left-images";     # edit this if needed
DIRECTORY_TO_CYCLE_RIGHT = "right-images";   # edit this if needed

# resolve full directory location of data set for left / right images

FULL_PATH_DIRECTORY_LEFT =  os.path.join(MASTER_PATH_TO_DATASET, DIRECTORY_TO_CYCLE_LEFT);
FULL_PATH_DIRECTORY_RIGHT =  os.path.join(MASTER_PATH_TO_DATASET, DIRECTORY_TO_CYCLE_RIGHT);

# get a list of the left image files and sort them (by timestamp in filename)

LEFT_FILE_LIST = sorted(os.listdir(FULL_PATH_DIRECTORY_LEFT));

# set this to a file timestamp to start from (empty is first example - outside lab)
# e.g. set to 1506943191.487683 for the end of the Bailey, just as the vehicle turns

SKIP_FORWARD_FILE_PATTERN = "1506942753.476428" # set to timestamp to skip forward to

CROP_DISPARITY = False; # display full or cropped disparity image
pause_playback = False; # pause until key press after each image

#####################################################################

disparity_mask = cv2.imread("mask2.png", cv2.IMREAD_GRAYSCALE)
canny_mask = cv2.imread("canny_mask.png", cv2.IMREAD_GRAYSCALE)

max_disparity = 128;

for filename_left in LEFT_FILE_LIST:

    # skip forward to start a file we specify by timestamp (if this is set)

    if ((len(SKIP_FORWARD_FILE_PATTERN) > 0) and not (SKIP_FORWARD_FILE_PATTERN in filename_left)):
        continue;
    elif ((len(SKIP_FORWARD_FILE_PATTERN) > 0) and (SKIP_FORWARD_FILE_PATTERN in filename_left)):
        SKIP_FORWARD_FILE_PATTERN = "";

    # from the left image filename get the corresponding right image

    filename_right = filename_left.replace("_L", "_R");
    full_path_filename_left = os.path.join(FULL_PATH_DIRECTORY_LEFT, filename_left);
    full_path_filename_right = os.path.join(FULL_PATH_DIRECTORY_RIGHT, filename_right);

    # for sanity print out these filenames

    print(full_path_filename_left);
    print(full_path_filename_right);
    print();

    # check the file is a PNG file (left) and check a corresponding right image
    # actually exists

    if ('.png' in filename_left) and (os.path.isfile(full_path_filename_right)):

        # read left and right images and display in windows
        # N.B. despite one being grayscale both are in fact stored as 3-channel
        # RGB images so load both as such

        imgL = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)
        # cv2.imshow('left image', imgL)

        imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR)
        # cv2.imshow('right image', imgR)

        print("-- files loaded successfully");
        print();

        left_right = np.vstack((imgL, imgR))
        left_right = cv2.resize(left_right, DEBUG_WINDOW_SIZE)
        # cv2.imshow('left and right', left_right)
        #
        canny = cv2.Canny(imgL, 100, 300, L2gradient=True)
        canny = cv2.dilate(canny,np.ones((3,3)))
        canny = cv2.bitwise_and(canny, cv2.bitwise_not(canny_mask))
        canny = cv2.bitwise_not(canny)


        (disparity, disparity_scaled) = stereo_disparity.calculate_disparity(imgL, imgR,
                                                                             CROP_DISPARITY, False)

        _, mask = cv2.threshold(disparity, 1, 1, cv2.THRESH_BINARY_INV)

        mask = np.uint8(mask)
        # disparity = np.uint8(disparity)

        # mask = disparity[np.where(disparity == -1)]
        # mask = cv2.equalizeHist(mask)

        disparity_in = cv2.inpaint(disparity, mask, 15, cv2.INPAINT_NS)

        disparity = cv2.bitwise_and(disparity,disparity_mask)
        # disparity_in = cv2.bitwise_and(disparity_in,canny)
        disparity_in = cv2.bitwise_and(disparity_in, disparity_mask)

        # cv2.imwrite("disparity.png",disparity)
        # cv2.merge((mask,mask,mask))


        points = stereo_to_3d.project_disparity_to_3d(disparity_in, max_disparity);


        plane, consensus = ransac.ransac(points, 1000)


        # write to file in an X simple ASCII X Y Z format that can be viewed in 3D
        # usin
        # uploading, selecting X Y Z format, press render , rotating the view)
        #
        # point_cloud_file = open('3d_points.txt', 'w');
        # csv_writer = csv.writer(point_cloud_file, delimiter=' ');
        # csv_writer.writerows(points);
        # point_cloud_file.close();

        # select a random subset of the 3D points (4 in total)
        # and them project back to the 2D image (as an example)

        pts = stereo_to_3d.project_3D_points_to_2D_image_points(consensus)
        plane_pts =  stereo_to_3d.project_3D_points_to_2D_image_points(plane)

        consensus_image = np.uint8(np.zeros(disparity_in.shape))
        for (y, x) in pts:
            consensus_image[y, x] = 255

        consensus_image = cv2.morphologyEx(consensus_image, cv2.MORPH_CLOSE, np.ones((3,3)))
        consensus_image = cv2.erode(consensus_image,np.ones((3,3)), iterations=3)

        consensus_image = cv2.bitwise_and(consensus_image, canny)


        _, contours, hierarchy = cv2.findContours(consensus_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if (len(contours) > 0):
            biggest_contour = contours[0]
            for c in contours:
                if (cv2.contourArea(c) > cv2.contourArea(biggest_contour)):
                    biggest_contour = c



            biggest_contour = cv2.convexHull(biggest_contour)
            epsilon = 0.01 * cv2.arcLength(biggest_contour, True)
            biggest_contour = cv2.approxPolyDP(biggest_contour, epsilon, True)





            # hull = cv2.convexHull(pts)

            # plane = cv2.drawContours(imgL, pts, -1, (0, 255, 0), 3)


            #
            # for (y,x) in pts:
            #     imgL[y,x] = [0,255,0]
            #
            # for i in range(len(plane_pts)):
            #     (x, y) = plane_pts[i]
            #     plane_pts[i] = (y, x)
            # plane_pts1 = np.array(plane_pts, np.int32);
            # plane_pts1 = plane_pts1.reshape((-1, 1, 2));
            #
            # cv2.polylines(imgL, [plane_pts1], True, (0, 255, 255), 3);

            imgL = cv2.drawContours(imgL, [biggest_contour], -1, (0, 255, 0), 3)


            imgL = cv2.resize(imgL, DEBUG_WINDOW_SIZE)
            canny = cv2.resize(canny, DEBUG_WINDOW_SIZE)
            canny = cv2.merge((canny, canny, canny))
            disparity = cv2.resize(cv2.equalizeHist(disparity), DEBUG_WINDOW_SIZE)
            disparity = cv2.merge((disparity,disparity,disparity))
            disparity_in = cv2.resize(cv2.equalizeHist(disparity_in), DEBUG_WINDOW_SIZE)
            disparity_in = cv2.merge((disparity_in, disparity_in, disparity_in))
            mask = cv2.resize(cv2.equalizeHist(mask), DEBUG_WINDOW_SIZE)
            mask = cv2.merge((mask, mask, mask))
            consensus_image = cv2.resize(cv2.equalizeHist(consensus_image), DEBUG_WINDOW_SIZE)
            consensus_image = cv2.merge((consensus_image, consensus_image, consensus_image))

            row1 = np.hstack((imgL, canny, disparity))
            row2 = np.hstack((disparity_in, mask, consensus_image))
            display_image = np.vstack((row1,row2))

            cv2.imshow("res", display_image)
            # keyboard input for exit (as standard), save disparity and cropping
            # exit - x
            # save - s
            # crop - c
            # pause - space

        key = cv2.waitKey(1 * (not (pause_playback))) & 0xFF;  # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)
        if (key == ord('x')):  # exit
            break;  # exit
        elif (key == ord('s')):  # save
            cv2.imwrite("sgbm-disparty.png", disparity_scaled);
            cv2.imwrite("left.png", imgL);
            cv2.imwrite("right.png", imgR);
        elif (key == ord('c')):  # crop
            CROP_DISPARITY = not (CROP_DISPARITY);
        elif (key == ord(' ')):  # pause (on next frame)
            pause_playback = not (pause_playback);
    else:
        print("-- files skipped (perhaps one is missing or not PNG)");
        print();

# close all windows

cv2.destroyAllWindows()

#####################################################################
