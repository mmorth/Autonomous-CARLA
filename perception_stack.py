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

import math
import numpy as np


##########################################################################################
# Object Distance Measurement ############################################################
##########################################################################################


# Draws the perception output on the current frame
def visualize_perception_output(img, depth_img, ss_img):
    # TODO: Create a method to visualize the perception output on the image
    pass


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
    object_roi_distance = pixel_distances[detection[1]:detection[3], detection[0]:detection[1]]

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
