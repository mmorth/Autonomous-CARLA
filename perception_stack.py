

'''
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


