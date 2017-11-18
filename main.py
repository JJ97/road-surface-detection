import cv2
import os, time
import numpy as np
import stereo_disparity, stereo_to_3d
import skip_locations, preprocessing, ransac,  postprocessing

# where is the data ? - set this to where you have it

MASTER_PATH_TO_DATASET = "/volumes/RAM Disk2/dataset/"  # ** need to edit this **
SKIP_FORWARD_FILE_PATTERN = skip_locations.CHURCH_STREET # set to timestamp to skip forward to

VIDEO_FILENAME = ""
SAVE_TO_VIDEO = VIDEO_FILENAME != ""

DISPARITY_MASK = cv2.imread("disparity_mask.png", cv2.IMREAD_GRAYSCALE)
CANNY_MASK = cv2.imread("canny_mask.png", cv2.IMREAD_GRAYSCALE)

DIRECTORY_TO_CYCLE_LEFT = "left-images"
DIRECTORY_TO_CYCLE_RIGHT = "right-images"
FULL_PATH_DIRECTORY_LEFT =  os.path.join(MASTER_PATH_TO_DATASET, DIRECTORY_TO_CYCLE_LEFT)
FULL_PATH_DIRECTORY_RIGHT =  os.path.join(MASTER_PATH_TO_DATASET, DIRECTORY_TO_CYCLE_RIGHT)
LEFT_FILE_LIST = sorted(os.listdir(FULL_PATH_DIRECTORY_LEFT))

DEBUG_WINDOW_SIZE = (300,300)

MAX_DISPARITY = 128

#####################################################################

def process_single_frame(full_path_filename_left, full_path_filename_right):
    imgL = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)
    imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR)

    # invariant = preprocessing.convert_illumination_invariant(imgL, alpha=0.42)

    green_mask = preprocessing.get_green_mask(imgL)
    consensus_mask = preprocessing.mask_canny_for_post_processing(imgL, CANNY_MASK)
    disparity = stereo_disparity.calculate_disparity(imgL, imgR)
    disparity_clean, holes_in_disparity = preprocessing.cleanup_disparity(disparity, DISPARITY_MASK)

    points = stereo_to_3d.project_disparity_to_3d(disparity_clean, MAX_DISPARITY)
    consensus_points_3d = ransac.ransac(points, iterations=1000)
    consensus_points_2d = stereo_to_3d.project_3D_points_to_2D_image_points(consensus_points_3d)

    consensus_image = postprocessing.create_consensus_image(consensus_points_2d, disparity_clean.shape)
    consensus_clean = postprocessing.cleanup_consensus(consensus_image, consensus_mask, green_mask)
    final_result = postprocessing.draw_detected_road_region(imgL, consensus_clean)

    hsv = cv2.cvtColor(imgL, cv2.COLOR_BGR2HSV)
    s = hsv[:,:,1]
    if (final_result is not None):
        display_debug_image(final_result, consensus_mask, s,
                            disparity_clean, green_mask, consensus_image)
        if (SAVE_TO_VIDEO):
            video.write(final_result)


def display_debug_image(final_result, consensus_mask, invariant,
                        disparity_clean, consensus_clean, consensus_image):
    final_result = cv2.resize(final_result, DEBUG_WINDOW_SIZE)
    consensus_mask = cv2.resize(consensus_mask, DEBUG_WINDOW_SIZE)
    consensus_mask = cv2.merge((consensus_mask, consensus_mask, consensus_mask))
    invariant = cv2.resize(cv2.equalizeHist(invariant), DEBUG_WINDOW_SIZE)
    invariant = cv2.merge((invariant, invariant, invariant))
    disparity_clean = cv2.resize(cv2.equalizeHist(disparity_clean), DEBUG_WINDOW_SIZE)
    disparity_clean = cv2.merge((disparity_clean, disparity_clean, disparity_clean))
    consensus_clean = cv2.resize(cv2.equalizeHist(consensus_clean), DEBUG_WINDOW_SIZE)
    consensus_clean = cv2.merge((consensus_clean, consensus_clean, consensus_clean))
    consensus_image = cv2.resize(cv2.equalizeHist(consensus_image), DEBUG_WINDOW_SIZE)
    consensus_image = cv2.merge((consensus_image, consensus_image, consensus_image))

    row1 = np.hstack((final_result, consensus_mask, invariant))
    row2 = np.hstack((disparity_clean, consensus_clean, consensus_image))
    display_image = np.vstack((row1, row2))

    cv2.imshow("res", display_image)

if (SAVE_TO_VIDEO):
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    video = cv2.VideoWriter(VIDEO_FILENAME,fourcc, 3.0, (1024,544))

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
        print(filename_left)
        process_single_frame(full_path_filename_left, full_path_filename_right)
        key = cv2.waitKey(40) & 0xFF  # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)
    else:
        print("-- files skipped (perhaps one is missing or not PNG)")
        print()

# close all windows
cv2.destroyAllWindows()
if (SAVE_TO_VIDEO):
    video.release()
#####################################################################
