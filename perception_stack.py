'''
NOTE: This file is a Work in Progress and such may not run correctly if at all. Until complete, please use perception_algorithm.py

This file contains the code for supporting a real-time CARLA simulation environment perception stack. It includes
1. Bounding box object detection using trained NN model
2. Object distance information from disparity image
3. Lane line and drivable space detection using Hough Lines
The below perception features are planned:
1. Detected object speed estimation using optical flow
2. Vehicle state estimation and localization using Visual Odometry
3. Vehicle 3D environment mapping using 3D camera FOV and image stitching
The below AV features are planned (not part of perception stack):
1. Vehicle control modeling using Kinematic and Dynamic Lateral/Longitudnal models
2. State Estimation and Localization using Kalman Filter
3. Mapping and Path Planning using state machines, maps, and graphs
'''

# Python Library/Package Imports
import cv2
import math
import numpy as np

# Project File Imports
from object_detection_nn import CLASSES, COLORS


##########################################################################################
# Object Distance Measurement ############################################################
##########################################################################################

# TODO: Determine where/whether to keep global variables
MIN_CONFIDENCE = 0.8

# Draws the perception output on the current frame
# Source: https://www.pyimagesearch.com/2021/08/02/pytorch-object-detection-with-pre-trained-networks/
def visualize_predictions(orig, depth_img, ss_img, detections, fov):
    # Detect and compute the x and y world coordinates from the depth map
    x_world, y_world = __compute_xzy_from_depth(depth_img, fov)

    global MIN_CONFIDENCE
    global CLASSES
    global COLORS
    # loop over the detections
    print("Found " + str(len(detections["boxes"])) + " detections!")
    for i in range(0, len(detections["boxes"])):
        # Extract the confidence (i.e., probability) associated with the prediction
        confidence = detections["scores"][i]
        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > MIN_CONFIDENCE:
            # extract the index of the class label from the detections
            idx = int(detections["labels"][i])
            # if label is not in classes, set it to other label
            if idx not in CLASSES:
                idx = 25

            # compute the (x, y)-coordinates of the bounding box for the object
            box = detections["boxes"][i].detach().cpu().numpy()
            (startX, startY, endX, endY) = box.astype("int")
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            
            # compute distance to the detected objects
            distance = 1000 * compute_object_distance(box, x_world, y_world, depth_img)
            label_distance = "{:.2f}m".format(distance)

            # draw the bounding box and label on the image
            cv2.rectangle(orig, (startX, startY), (endX, endY), COLORS[idx-1], 2)

            # display the prediction to our terminal and on the image
            print("[INFO] {}".format(label))
            print("[INFO] {}".format(label_distance))

            # compute the text locations on the image
            y = startY - 15 if startY - 15 > 15 else startY + 15
            yd = startY - 30 if startY - 30 > 30 else startY + 30
            cv2.putText(orig, label, (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx-1], 2)
            cv2.putText(orig, label_distance, (startX, yd),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx-1], 2)

    # show the output image
    cv2.imshow("CameraRGB", orig)
    cv2.waitKey(1)


##########################################################################################
# Object Distance Measurement ############################################################
##########################################################################################

# TODO: Learn about Camera Calibration parameters f, cu, cv and why formulas are the way they are

# This helper function obtains the world xyz coordinates from the input image
def __compute_xzy_from_depth(depth, fov):
    # Compute the camera intrinsic calibration matrix   
    img_height, img_width = depth.shape
    
    # TODO: Use the updated camera calibration method and parameters instead of perfect result
    z = depth[:,:,]
    # Focal length measures how strongly the camera converges or diverges light (optical power)
    # Since we are working with square pixels, fu=fv
    f = img_width / (2 * math.tan(fov * math.pi / 360))
    # cu, cv specify the intersection of the optical axis with the image plane
    cu = img_width / 2
    cv = img_height / 2

    # Vectorize the image plane coordinate points for faster computation
    u, v = np.meshgrid(np.arange(1, img_width + 1, 1), np.arange(1, img_height + 1, 1))

    # Compute the x, y, z world coordinates (inverse projection)
    # Each variable represents the input u/v required to convert to the equivalent world axis coordinate
    x = ((u - cu) * z) / f
    y = ((v - cv) * z) / f

    return x, y, z


# This function computes the minimum point of impact to a specified object
def __compute_min_distance_to_impact(detection, x, y, z):
    # Compute a 2D strcuture that contains the distance of each image pixel in meters
    pixel_distances = np.sqrt(x**2 + y**2 + z**2)

    # Compute the region roi of the object using bouding box coordinates of the detection
    object_roi_distance = pixel_distances[detection[1]:detection[3], detection[0]:detection[2]]

    # Compute and return the minimum distance from the roi
    return np.min(object_roi_distance)


# Computes and returns the distance to the object
def compute_object_distance(depth_img, detection, fov):
    # TODO: Compute this once for every frame instead of for every object
    # Find the x, y, and z world coordinates from the depth image
    x, y, z = __compute_xzy_from_depth(depth_img, fov)

    # Compute and return the min distance to object collision
    return __compute_min_distance_to_impact(detection, x, y, z)


##########################################################################################
# Compute/Find Lane Lines ################################################################
##########################################################################################


# Finds and returns the lane lines
def find_lane_lines(ss_img):
    pass


##########################################################################################
# Compute Object Speed/Trajectory ########################################################
##########################################################################################


# Computes and returns the specfied object's speed and trajectory
def compute_object_speed_trajectory(img, depth_img, region):
    pass


##########################################################################################
# Visual Odometry ########################################################################
##########################################################################################


# Computes visual odometry for the object
def visual_odometry(img):
    pass


##########################################################################################
# 3D Map Environment #####################################################################
##########################################################################################


# Computes the 3D scene map of the current environment
def environment_map_3d(img, depth_img, ss_img):
    pass
