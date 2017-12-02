import cv2
import os, time, math, random
import numpy as np
import stereo_disparity, stereo_to_3d
import skip_locations, preprocessing, ransac,  postprocessing

MASTER_PATH_TO_DATASET = "dataset"
DIRECTORY_TO_CYCLE_LEFT = "left-images"
DIRECTORY_TO_CYCLE_RIGHT = "right-images"
FULL_PATH_DIRECTORY_LEFT =  os.path.join(MASTER_PATH_TO_DATASET, DIRECTORY_TO_CYCLE_LEFT)
FULL_PATH_DIRECTORY_RIGHT =  os.path.join(MASTER_PATH_TO_DATASET, DIRECTORY_TO_CYCLE_RIGHT)
LEFT_FILE_LIST = sorted(os.listdir(FULL_PATH_DIRECTORY_LEFT))

# set to timestamp to skip forward to
# convenient locations available as variables in skip_locations
SKIP_FORWARD_FILE_PATTERN = skip_locations.BEGINNING
# images are resized by a factor of DOWNSAMPLE_RATE in range 0->1 before processing
# sacrifices some accuracy for faster performance
DOWNSAMPLE_RATE = 1
# crop the top 150 pixels for faster processing since they will (hopefully) never be road
SKY_CROP_HEIGHT = int(150*DOWNSAMPLE_RATE)

VIDEO_FILENAME = "" #filename for saving results to video
SAVE_TO_VIDEO = (VIDEO_FILENAME != "") # only save to video if filename specified

# load predefined mask for disparity then downscale and crop
# used to select a generic ROI for any frame i.e. ignore sky and car bonnet
DISPARITY_MASK = cv2.imread("disparity_mask.png", cv2.IMREAD_GRAYSCALE)
DISPARITY_MASK = cv2.resize(DISPARITY_MASK, (0, 0), fx=DOWNSAMPLE_RATE, fy=DOWNSAMPLE_RATE)[SKY_CROP_HEIGHT:,]

# load predefined mask for canny then downscale and crop
# used to prevent road markings from splitting consensus in half during postprocessing
CANNY_MASK = cv2.imread("canny_mask.png", cv2.IMREAD_GRAYSCALE)
CANNY_MASK = cv2.resize(CANNY_MASK, (0, 0), fx=DOWNSAMPLE_RATE, fy=DOWNSAMPLE_RATE)[SKY_CROP_HEIGHT:,]

IMG_SIZE = (1024,544) #size of our input images
SAVE_VIDEO_SIZE = (int(1024/2), int(544)) #size of each frame in the output video
MAX_DISPARITY = 128

def process_single_frame(full_path_filename_left, full_path_filename_right, previous_plane):
    """ Load, identify and draw road region on a single pair of left, right images

    Arguments:
        full_path_filename_left  -- path to image from left camera
        full_path_filename_right -- path to image from right camera
        previous_plane           -- best plane estimate from previous frame
                                    averaged against RANSAC of current frame to improve results
    """
    imgL = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)
    imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR)

    ##########################################
    # PRE-PROCESSING
    ##########################################

    # downsample left and right images for speed / accuracy tradeoff
    imgL, imgR = preprocessing.downsample(imgL, imgR, DOWNSAMPLE_RATE)
    # crop SKY_CROP_HEIGHT pixels out of both images since roads will (hopefully) never be that high up
    left_cropped, right_cropped = preprocessing.crop_sky(imgL, imgR, SKY_CROP_HEIGHT)
    # threshold for pixels within a certain range of 'green' in HSV
    # used to ignore disparity points formed from grass, bushes etc.
    green_mask = preprocessing.get_green_mask(left_cropped)
    # create a mask from canny of left image, ignoring areas directly in front of the car
    # used to separate consensus points if there are strong edges e.g. sides of roads going through them
    edge_mask = preprocessing.create_edge_mask(left_cropped, CANNY_MASK)
    # create disparity using code provided on DUO
    disparity = stereo_disparity.calculate_disparity(left_cropped, right_cropped)
    # improve disparity results before further processing i.e. filtering noise, infilling and masking ROI
    disparity_clean, holes_in_disparity = preprocessing.cleanup_disparity(disparity, DISPARITY_MASK, green_mask)


    ##########################################
    # 3D RANSAC
    ##########################################

    # convert disparity image to a list of 3D points using code provided on DUO
    points = stereo_to_3d.project_disparity_to_3d(disparity_clean, MAX_DISPARITY)
    # detect road region using RANSAC
    normal, plane, consensus_points_3d = ransac.ransac(points, previous_plane)

    # if a road region has been found
    if (normal is not None):
        # convert road region points back into 2D image points using code provided on DUO
        consensus_points_2d = stereo_to_3d.project_3D_points_to_2D_image_points(consensus_points_3d)

        # get central point of consensus image and plot normal vector from there
        centroid = np.mean(consensus_points_3d, axis=0)
        normal_vector_points_3d = [centroid, centroid - normal] # subtract since +ve y points down
        normal_vector_points_2d = stereo_to_3d.project_3D_points_to_2D_image_points(normal_vector_points_3d)

        ##########################################
        # POST-PROCESSING
        ##########################################

        # convert list of 2d road region points into an actual image for further processing
        consensus_image = postprocessing.create_consensus_image(consensus_points_2d, disparity_clean.shape)
        # refine consensus using mask derived from canny and morphology functions
        consensus_clean = postprocessing.cleanup_consensus(consensus_image, edge_mask)
        # draw convex hull of largest consensus region onto left image for display
        final_result, centroid = postprocessing.draw_detected_road_region(left_cropped, consensus_clean)


        # draw normal vector in center of largest consensus region
        final_result = postprocessing.draw_normal_vector(final_result, normal_vector_points_2d, centroid)

        # add sky back in to image and resize to original dimensions
        final_result_uncropped = postprocessing.uncrop(final_result, imgL, SKY_CROP_HEIGHT)
        final_result_uncropped = cv2.resize(final_result_uncropped, IMG_SIZE)

        # create single image of results and disparity for display
        disparity_display = cv2.resize(cv2.cvtColor(cv2.equalizeHist(disparity_clean), cv2.COLOR_GRAY2BGR), IMG_SIZE)
        save_image = np.vstack((final_result_uncropped, disparity_display))
        save_image = cv2.resize(save_image, SAVE_VIDEO_SIZE)
        cv2.imshow("results", save_image)

        # write results to video file
        if (SAVE_TO_VIDEO):
            video.write(save_image)

        return normal, plane

    # if no road region detected
    else:
        return np.array([0,0,0]), []

# initialise a VideoWriter object for saving results
if (SAVE_TO_VIDEO):
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    video = cv2.VideoWriter(VIDEO_FILENAME,fourcc, 3.0, SAVE_VIDEO_SIZE)

plane = []
for filename_left in LEFT_FILE_LIST:
    # skip forward to start a file we specify by timestamp (if this is set)
    if ((len(SKIP_FORWARD_FILE_PATTERN) > 0) and not (SKIP_FORWARD_FILE_PATTERN in filename_left)):
        continue
    elif ((len(SKIP_FORWARD_FILE_PATTERN) > 0) and (SKIP_FORWARD_FILE_PATTERN in filename_left)):
        SKIP_FORWARD_FILE_PATTERN = ""

    # from the left image filename get the corresponding right image
    filename_right = filename_left.replace("_L", "_R")
    full_path_filename_left = os.path.join(FULL_PATH_DIRECTORY_LEFT, filename_left)
    full_path_filename_right = os.path.join(FULL_PATH_DIRECTORY_RIGHT, filename_right)

    # check the file is a PNG file (left) and check a corresponding right image
    # actually exists
    if ('.png' in filename_left) and (os.path.isfile(full_path_filename_right)):
        # get road region normal and corresponding plane for this pair of images
        normal, plane = process_single_frame(full_path_filename_left, full_path_filename_right, plane)
        # map normal vector to a string for display
        normal_str = ', '.join(map(str,normal))
        # output filename and normal vector as specified
        print(filename_left)
        print("{} : road surface normal ({})".format(filename_right, normal_str))

        key = cv2.waitKey(40) & 0xFF  # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)

# close all windows
cv2.destroyAllWindows()
# release video file handle
if (SAVE_TO_VIDEO):
    video.release()
