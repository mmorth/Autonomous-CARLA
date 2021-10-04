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

# TODO: Add helper functions for perception detection

# Draws the perception output on the current frame
def visualize_perception_output(img, depth_img, ss_img):
    pass


# Computes and returns the distance to the object
def compute_object_distance(depth_img, region):
    pass


# Finds and returns the lane lines
def find_lane_lines(ss_img):
    pass


# Computes and returns the specfied object's speed and trajectory
def compute_object_speed_trajectory(img, depth_img, region):
    pass


# Computes visual odometry for the object
def visual_odometry(img):
    pass


# Computes the 3D scene map of the current environment
def environment_map_3d(img, depth_img, ss_img):
    pass
