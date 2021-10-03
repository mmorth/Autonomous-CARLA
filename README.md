# Autonomous-CARLA
This repo contains experimental code for the CARLA Simulation Environment. The supported features are below:
* Collect sensor data from the CARLA simulation environment (data_collector.py)
* Auto label bounding boxes and labels for objects in the CARLA simulation environment (auto_labeler.py)
* Train a PyTorch neural network to detect the bounding box locations of vehicles, pedestrains, and traffic signs in the CARLA simulation environment (object_detection_nn.py)
* Run real-time bounding box object detection, depth estimation, and lane line/drivable space detection in the CARLA simulation environment (perception_algorithm.py will become perception_stack.py)

## Requirements
* Install Python: [https://www.python.org/downloads/](https://www.python.org/downloads/)
* Install Carla: [https://carla.readthedocs.io/en/latest/start_quickstart/](https://carla.readthedocs.io/en/latest/start_quickstart/)
* Install PyTorch: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

## Quickstart Guide
This quickstart guide will walk through working with all the major features of this repo (assumes requirements are met):
1. Starting the CARLA Simulation Environment Server:
    - In a command prompt, navigate to the directory containing CARLA.exe and run the below command:
        ```cmd
        CarlaUE4.exe -windowed -carla-server
        ```
2. Collecting Sensor Data From CARLA
    - Navigate to the root of this repo, and run data_collector.py to collect sensor data from CARLA (modify the global variables to control number of episodes and frames to capture) using the following command:
        ```cmd
        python data_collector.py --autopilot --images-to-disk
        ```
3. Auto-label Object Bounding Boxes:
     - Run auto_labeler.py to generate JSON files with object labels and bounding box coordinates from the captured data_collection images (update object class labels to detect if desired) using the following command:
        ```cmd
        python auto_labeler.py
        ```
4. Train Bounding Box Objection Detection Neural Network Model:
    - Run object_detection_nn.py on the captured and labeled images to train a neural network to detect object bounding box locations (modify the global variables as desired) using the following command:
        ```cmd
        python object_detection_nn.py
        ```
5. Real-time Perception Stack:
    - Run perception_algorithm.py to run real-time object detection, distance estimation, and lane line detection using the following command:
        ```cmd
        python perception_algorithm.py
        ```

## Detailed Feature Description
This section describes the features of this repo as well as future features.

### Perception Features
#### Data Collection, Labeling, and Training:
* CARLA sensor data collection using the CARLA API
* Auto labeling CARLA object's bounding boxes and class using the semantic segmentation image
* Customizable Transfer Learning from VGG16 COCO pre-trained model using PyTorch
#### Implemented Perception Stack:
The following perception stack items can be executed in real-time detection
* Object bounding box location from trained model
* Distance estimation of detected objects using the disparity image
* Lane detection using lane thresholding and Hough Transform
#### Future Perception Stack:
* Vehicle speed and trajectory estimation using Optical Flow
* Vehicle localization using Visual Odometry
* 3D, 360 Degree Environment Map using Image Stitching
* Improvements to customizability of perception stack (to be defined)

### Controls Features
Future work is planned to create a control model for use in this Autonomous CARLA stack.
* Kinematic, Dynamic, Longitudnal, and Lateral Control Models

### State Estimation and Localization Features
Future work is planned to create a state estimation and localization algorithm for use in this Autonomous CARLA stack.
* State estimation using variations of the Kalman Filter and vehicle sensors

### Mapping and Path Planning
Future work is planned to create a mapping and path planning algorithm for use in this Autonomous CARLA stack.
* Vehicle mapping and path planning using graphs and state machines

## Resrouces and References
Below are links to code and theory references used in this project:
* Visual Perception for Self-Driving Cars Coursera Course by University of Toronto: https://www.coursera.org/learn/visual-perception-self-driving-cars
* CARLA Documentation, Examples, and API: https://carla.org/
