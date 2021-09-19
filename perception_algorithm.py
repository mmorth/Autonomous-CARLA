#!/usr/bin/env python3

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Basic CARLA client example."""

from __future__ import print_function

import argparse
from carla_nn import CLASSES, COLORS
import cv2
from numpy.testing._private.utils import measure
import keyboard
import logging
import math
import os
import random
import time
import torch

import numpy as np

from carla.client import make_carla_client
from carla.image_converter import to_bgra_array, depth_to_array
from carla.sensor import Camera, Lidar
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line

from carla.client import make_carla_client, VehicleControl

import transforms as T

# TODO: Update into a Python class
# State variables 
reverse_on = False
enable_autopilot = False

# Constants
MIN_CONFIDENCE = 0.8
TRAIN_TEST_SPLIT = 0.8
IMAGE_DIR = "_out"
GROUND_TRUTH_DIR = "GroundTruthRGB"
LABEL_DIR = "Objects"
FRAMES_PER_EPISODE = 1000


# Receives the keyboard inputs from the user
def get_keyboard_control():
    """
    Return a VehicleControl message based on the pressed keys. Return None
    if a new episode was requested.
    """
    control = VehicleControl()
    if keyboard.is_pressed('a'):
        control.steer = -1.0
    if keyboard.is_pressed('d'):
        control.steer = 1.0
    if keyboard.is_pressed('w'):
        control.throttle = 1.0
    if keyboard.is_pressed('s'):
        control.brake = 1.0
    # if keys[K_SPACE]:
    #     control.hand_brake = True
    if keyboard.is_pressed('q'):
        global reverse_on
        reverse_on = not reverse_on
    if keyboard.is_pressed('p'):
        global enable_autopilot
        enable_autopilot = not enable_autopilot
    control.reverse = reverse_on
    return control


# Source: https://www.coursera.org/learn/visual-perception-self-driving-cars
def compute_plane(xyz):
    """
    Computes plane coefficients a,b,c,d of the plane in the form ax+by+cz+d = 0
    Arguments:
    xyz -- tensor of dimension (3, N), contains points needed to fit plane.
    k -- tensor of dimension (3x3), the intrinsic camera matrix
    Returns:
    p -- tensor of dimension (1, 4) containing the plane parameters a,b,c,d
    """
    ctr = xyz.mean(axis=1)
    normalized = xyz - ctr[:, np.newaxis]
    M = np.dot(normalized, normalized.T)

    p = np.linalg.svd(M)[0][:, -1]
    d = np.matmul(p, ctr)

    p = np.append(p, -d)

    return p


# Source: https://www.coursera.org/learn/visual-perception-self-driving-cars
def dist_to_plane(plane, x, y, z):
    """
    Computes distance between points provided by their x, and y, z coordinates
    and a plane in the form ax+by+cz+d = 0
    Arguments:
    plane -- tensor of dimension (4,1), containing the plane parameters [a,b,c,d]
    x -- tensor of dimension (Nx1), containing the x coordinates of the points
    y -- tensor of dimension (Nx1), containing the y coordinates of the points
    z -- tensor of dimension (Nx1), containing the z coordinates of the points
    Returns:
    distance -- tensor of dimension (N, 1) containing the distance between points and the plane
    """
    a, b, c, d = plane

    return (a * x + b * y + c * z + d) / np.sqrt(a**2 + b**2 + c**2)


def ransac_plane_fit(xyz_data):
    """
    Computes plane coefficients a,b,c,d of the plane in the form ax+by+cz+d = 0
    using ransac for outlier rejection.

    Arguments:
    xyz_data -- tensor of dimension (3, N), contains all data points from which random sampling will proceed.
    num_itr -- 
    distance_threshold -- Distance threshold from plane for a point to be considered an inlier.

    Returns:
    p -- tensor of dimension (1, 4) containing the plane parameters a,b,c,d
    """
    
    # Set thresholds:
    num_itr = 100  # RANSAC maximum number of iterations
    min_num_inliers = 300  # RANSAC minimum number of inliers
    distance_threshold = 0.01  # Maximum distance from point to plane for point to be considered inlier
    max_inliers = 0
    max_inliers_indices = 0
    
    for i in range(num_itr):
        # Step 1: Choose a minimum of 3 points from xyz_data at random.
        chosen = np.random.choice(xyz_data.shape[1], 3, replace=False)
        points = xyz_data[:, chosen]
        
        # Step 2: Compute plane model
        plane_model = compute_plane(points)
        
        # Step 3: Find number of inliers
        distance = dist_to_plane(plane_model, xyz_data[0, :].T, xyz_data[1, :].T, xyz_data[2, :].T)   
        inliers = len(distance[distance > distance_threshold])
        
        # Step 4: Check if the current number of inliers is greater than all previous iterations and keep the inlier set with the largest number of points.
        if inliers > max_inliers:
            max_inliers = inliers
            max_inliers_indices = np.where(distance < distance_threshold)[0]

        # Step 5: Check if stopping criterion is satisfied and break.         
        if inliers > min_num_inliers:
            break
        
    # Step 6: Recompute the model parameters using largest inlier set.         
    output_plane = compute_plane(xyz_data[:, max_inliers_indices])  
    
    return output_plane


def estimate_lane_lines(segmentation_output):
    """
    Estimates lines belonging to lane boundaries. Multiple lines could correspond to a single lane.

    Arguments:
    segmentation_output -- tensor of dimension (H,W), containing semantic segmentation neural network output

    Returns:
    lines -- tensor of dimension (N, 4) containing lines in the form of [x_1, y_1, x_2, y_2], where [x_1,y_1] and [x_2,y_2] are
    the coordinates of two points on the line in the (u,v) image coordinate frame.
    """
    # Step 1: Create an image with pixels belonging to lane boundary categories from the output of semantic segmentation
    lane_boundary_mask = np.zeros(segmentation_output.shape).astype(np.uint8)
    lane_boundary_mask[segmentation_output==6] = 255
    lane_boundary_mask[segmentation_output==8] = 255

    # Step 2: Perform Edge Detection using cv2.Canny()
    lane_edges = cv2.Canny(lane_boundary_mask, 100, 150)

    # Step 3: Perform Line estimation using cv2.HoughLinesP()
    lanes = cv2.HoughLinesP(lane_edges, rho=10, theta=np.pi/180, threshold=200, minLineLength=150, maxLineGap=50)
    
    # Note: Make sure dimensions of returned lines is (N x 4)
    lanes = lanes.reshape((-1, 4))

    return lanes


# Source: https://www.coursera.org/learn/visual-perception-self-driving-cars
def get_slope_intecept(lines):
    slopes = (lines[:, 3] - lines[:, 1]) / (lines[:, 2] - lines[:, 0] + 0.001)
    intercepts = ((lines[:, 3] + lines[:, 1]) - slopes * (
        lines[:, 2] + lines[:, 0])) / 2
    return slopes, intercepts


# Graded Function: merge_lane_lines
def merge_lane_lines(lines):
    """
    Merges lane lines to output a single line per lane, using the slope and intercept as similarity measures.
    Also, filters horizontal lane lines based on a minimum slope threshold.

    Arguments:
    lines -- tensor of dimension (N, 4) containing lines in the form of [x_1, y_1, x_2, y_2],
    the coordinates of two points on the line.

    Returns:
    merged_lines -- tensor of dimension (N, 4) containing lines in the form of [x_1, y_1, x_2, y_2],
    the coordinates of two points on the line.
    """
    # Step 0: Define thresholds
    slope_similarity_threshold = 0.1
    intercept_similarity_threshold = 40
    min_slope_threshold = 0.3
    clusters = []
    current_inds = []
    itr = 0
    
    # Step 1: Get slope and intercept of lines
    slopes, intercepts = get_slope_intecept(lines)
    
    # Step 2: Determine lines with slope less than horizontal slope threshold.
    slopes_horizontal = np.abs(slopes) > min_slope_threshold

    # Step 3: Iterate over all remaining slopes and intercepts and cluster lines that are close to each other using a slope and intercept threshold.
    for slope, intercept in zip(slopes, intercepts):
        in_clusters = np.array([itr in current for current in current_inds])
        if not in_clusters.any():
            slope_cluster = np.logical_and(slopes < (slope+slope_similarity_threshold), slopes > (slope-slope_similarity_threshold))
            intercept_cluster = np.logical_and(intercepts < (intercept+intercept_similarity_threshold), intercepts > (intercept-intercept_similarity_threshold))
            inds = np.argwhere(slope_cluster & intercept_cluster & slopes_horizontal).T
            if inds.size:
                current_inds.append(inds.flatten())
                clusters.append(lines[inds])
        itr += 1
        
    # Step 4: Merge all lines in clusters using mean averaging
    merged_lines = [np.mean(cluster, axis=1) for cluster in clusters]
    
    # Note: Make sure dimensions of returned lines is (N x 4)
    merged_lines = np.array(merged_lines).reshape((-1, 4))

    return merged_lines


# Source: https://www.coursera.org/learn/visual-perception-self-driving-cars
def extrapolate_lines(lines, y_min, y_max):
    slopes, intercepts = get_slope_intecept(lines)

    new_lines = []

    for slope, intercept, in zip(slopes, intercepts):
        x1 = (y_min - intercept) / slope
        x2 = (y_max - intercept) / slope
        new_lines.append([x1, y_min, x2, y_max])

    return np.array(new_lines)


# Source: https://www.coursera.org/learn/visual-perception-self-driving-cars
def find_closest_lines(lines, point):
    x0, y0 = point
    distances = []
    for line in lines:
        x1, y1, x2, y2 = line
        distances.append(((x2 - x1) * (y1 - y0) - (x1 - x0) *
                          (y2 - y1)) / (np.sqrt((y2 - y1)**2 + (x2 - x1)**2)))

    distances = np.abs(np.array(distances))
    sorted = distances.argsort()

    return lines[sorted[0:2], :]


# Draws lane lines based on the semantic segmentation output
def draw_lane_lines_and_drivable_space(orig, depth_img, ss_img, fov):
    # TODO: Optimize the x, y, z computation for real-time running
    # Source: https://github.com/carla-simulator/carla/issues/56

    # Compute the camera intrinsic calibration matric
    img_height, img_width = np.shape(img)
    f = img_width / (2 * math.tan(fov * math.pi / 360))
    cu = img_width / 2
    cv = img_height / 2

    z = depth_img
    x = np.zeros((img_height, img_width))
    y = np.zeros((img_height, img_width))

    # Convert from the (x y) pixel coordinates to the (x, y, z) world coordinates
    for i in range(img_height):
        for j in range(img_width):
            x[i, j] = ((j+1 - cu)*z[i, j]) / f
            y[i, j] = ((i+1 - cv)*z[i, j]) / f

    # Get road mask by choosing pixels in segmentation output with value 7
    road_mask = np.zeros(ss_img.shape)
    road_mask[ss_img == 7] = 1

    # Get x,y, and z coordinates of pixels in road mask
    x_ground = x[road_mask == 1]
    y_ground = y[road_mask == 1]
    z_ground = depth_img[road_mask == 1]
    xyz_ground = np.stack((x_ground, y_ground, z_ground))

    # Estimate the ground plane (drivable space)
    drivable_space = ransac_plane_fit(xyz_ground)

    # Lane estimation
    lane_lines = estimate_lane_lines(ss_img)

    filtered_lane_lines = merge_lane_lines(lane_lines)

    max_y = orig.shape[0]
    min_y = np.min(np.argwhere(road_mask == 1)[:, 0])

    extrapolated_lanes = extrapolate_lines(filtered_lane_lines, max_y, min_y)

    # TODO: Determine how to compute the lane midpoint
    final_lane_lines = find_closest_lines(extrapolated_lanes, np.array[800, 900])

    # TODO: Draw the lane lines and show the drivable space image


    return x, y, final_lane_lines, drivable_space


# Computes the distance in meters from the detected objects
def find_min_distance_to_detection(detection, x, y, z):
    # Step 1: Compute distance of every pixel in the detection bounds
    x_min, y_min, x_max, y_max = detection.astype("int")
    box_x = x[y_min:y_max, x_min:x_max]
    box_y = y[y_min:y_max, x_min:x_max]
    box_z = z[y_min:y_max, x_min:x_max]
    box_distances = np.sqrt(box_x**2 + box_y**2 + box_z**2)
    
    # Step 2: Find minimum distance
    min_distance = np.min(box_distances)

    return min_distance


# Visualizes predictions
# Source: https://www.pyimagesearch.com/2021/08/02/pytorch-object-detection-with-pre-trained-networks/
def visualize_predictions(orig, depth_img, ss_img, detections, fov):
    global MIN_CONFIDENCE
    global CLASSES
    global COLORS
    # loop over the detections
    print("Found " + str(len(detections["boxes"])) + " detections!")
    for i in range(0, len(detections["boxes"])):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections["scores"][i]
        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > MIN_CONFIDENCE:
            # extract the index of the class label from the detections,
            # then compute the (x, y)-coordinates of the bounding box
            # for the object
            idx = int(detections["labels"][i])
            if idx not in CLASSES:
                idx = 25
            box = detections["boxes"][i].detach().cpu().numpy()
            (startX, startY, endX, endY) = box.astype("int")
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)

            # use semantic segmentation image to display the lane lines and drivable space
            x_world, y_world, lane_lines, drivable_space = draw_lane_lines_and_drivable_space(orig, depth_img, ss_img, fov)

            # compute distance to the detected objects
            distance = find_min_distance_to_detection(box, x_world, y_world, depth_img)
            label_distance = "{}m".format(distance)

            # display the prediction to our terminal
            print("[INFO] {}".format(label))
            print("[INFO] {}".format(label_distance))
            # draw the bounding box and label on the image
            cv2.rectangle(orig, (startX, startY), (endX, endY),
                COLORS[idx-1], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            yd = startY - 30 if startY - 30 > 30 else startY + 30
            cv2.putText(orig, label, (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx-1], 2)
            cv2.putText(orig, label_distance, (startX, yd),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx-1], 2)

    # show the output image
    cv2.imshow("CameraRGB", orig)
    cv2.waitKey(1)


def run_carla_client(args):
    # We assume the CARLA server is already waiting for a client to connect at
    # host:port. To create a connection we can use the `make_carla_client`
    # context manager, it creates a CARLA client object and starts the
    # connection. It will throw an exception if something goes wrong. The
    # context manager makes sure the connection is always cleaned up on exit.
    camera_fov = 0

    with make_carla_client(args.host, args.port) as client:
        print('CarlaClient connected')
        
        # TODO: Load object detection model
        model = torch.load(os.path.join(os.getcwd(), "models", 'model' + str(0) + '.pt'))
        model = model.cuda()
        print("Successfully loaded model!")

        # Start a new simulation environment
        if args.settings_filepath is None:

            # Create a CarlaSettings object. This object is a wrapper around
            # the CarlaSettings.ini file. Here we set the configuration we
            # want for the new episode.
            settings = CarlaSettings()
            settings.set(
                SynchronousMode=True,
                SendNonPlayerAgentsInfo=True,
                NumberOfVehicles=300,
                NumberOfPedestrians=300,
                WeatherId=0,
                QualityLevel=args.quality_level)
            settings.randomize_seeds()

            # Now we want to add a couple of cameras to the player vehicle.
            # We will collect the images produced by these cameras every
            # frame.

            # TODO: Replace the depth camera with a second RGB camera and manually compute depth

            # The default camera captures RGB images of the scene.
            camera0 = Camera('CameraRGB')
            # Set image resolution in pixels.
            camera0.set_image_size(800, 600)
            # Set its position relative to the car in meters.
            camera0.set_position(0.30, 0, 1.30)
            camera_fov = camera0.FOV
            settings.add_sensor(camera0)

            # Let's add another camera producing ground-truth depth.
            camera1 = Camera('CameraDepth', PostProcessing='Depth')
            camera1.set_image_size(800, 600)
            camera1.set_position(0.30, 0, 1.30)
            settings.add_sensor(camera1)

            camera2 = Camera('CameraSemanticSegmentation', PostProcessing='SemanticSegmentation')
            # Set image resolution in pixels.
            camera2.set_image_size(800, 600)
            # Set its position relative to the car in meters.
            camera2.set_position(0.30, 0, 1.30)
            settings.add_sensor(camera2)

        else:

            # Alternatively, we can load these settings from a file.
            with open(args.settings_filepath, 'r') as fp:
                settings = fp.read()

        # Now we load these settings into the server. The server replies
        # with a scene description containing the available start spots for
        # the player. Here we can provide a CarlaSettings object or a
        # CarlaSettings.ini file as string.
        scene = client.load_settings(settings)

        # Choose one player start at random.
        number_of_player_starts = len(scene.player_start_spots)
        player_start = random.randint(0, max(0, number_of_player_starts - 1))

        # Notify the server that we want to start the episode at the
        # player_start index. This function blocks until the server is ready
        # to start the episode.
        print('Starting new episode at %r...' % scene.map_name)
        client.start_episode(player_start)

        # Iterate every frame in the simulation
        frame = 0
        depth_img = np.zeros((800, 600, 1), dtype = "uint8")
        ss_img = np.zeros((800, 600, 1), dtype = "uint8")
        while True:
            frame += 1

            # Read the data produced by the server this frame.
            measurements, sensor_data = client.read_data()

            # Print some of the measurements.
            print_measurements(measurements)

            # TODO: View the captured images in real-time
            # Save the images to disk if requested.
            for name, measurement in sensor_data.items():
                if name == 'CameraRGB':
                    # Obtain and detect objects on the RGB image
                    img = to_bgra_array(measurement)
                    orig = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                    # TODO: detect object locations using trained model
                    transform = T.ToTensor()
                    input = transform(orig)[0]
                    img = input.unsqueeze_(0)
                    model.eval()
                    pred = model(input.cuda())[0]
                    visualize_predictions(orig, depth_img, ss_img, pred, camera_fov)
                elif name == 'CameraDepth':
                    # Obtain and save the depth measurements for distance visualization
                    depth_img = depth_to_array(measurement)
                elif name == 'CameraSemanticSegmentation':
                    # Obtain and save the semantic segmentation measurements for lane visualization
                    ss_img = to_bgra_array(measurement)

            # Send the position to the client
            control = get_keyboard_control()

            if enable_autopilot:
                client.send_control(measurements.player_measurements.autopilot_control)
            else:
                client.send_control(control)


def print_measurements(measurements):
    number_of_agents = len(measurements.non_player_agents)
    player_measurements = measurements.player_measurements
    message = 'Vehicle at ({pos_x:.1f}, {pos_y:.1f}), '
    message += '{speed:.0f} km/h, '
    message += 'Collision: {{vehicles={col_cars:.0f}, pedestrians={col_ped:.0f}, other={col_other:.0f}}}, '
    message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road, '
    message += '({agents_num:d} non-player agents in the scene)'
    message = message.format(
        pos_x=player_measurements.transform.location.x,
        pos_y=player_measurements.transform.location.y,
        speed=player_measurements.forward_speed * 3.6, # m/s -> km/h
        col_cars=player_measurements.collision_vehicles,
        col_ped=player_measurements.collision_pedestrians,
        col_other=player_measurements.collision_other,
        other_lane=100 * player_measurements.intersection_otherlane,
        offroad=100 * player_measurements.intersection_offroad,
        agents_num=number_of_agents)
    print_over_same_line(message)


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '-l', '--lidar',
        action='store_true',
        help='enable Lidar')
    argparser.add_argument(
        '-q', '--quality-level',
        choices=['Low', 'Epic'],
        type=lambda s: s.title(),
        default='Low',
        help='graphics quality level, a lower level makes the simulation run considerably faster.')
    argparser.add_argument(
        '-i', '--images-to-disk',
        action='store_true',
        dest='save_images_to_disk',
        help='save images (and Lidar data if active) to disk')
    argparser.add_argument(
        '-c', '--carla-settings',
        metavar='PATH',
        dest='settings_filepath',
        default=None,
        help='Path to a "CarlaSettings.ini" file')

    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    args.out_filename_format = '_out/episode_{:0>4d}/{:s}/{:0>4d}_{:0>6d}'

    while True:
        try:

            run_carla_client(args)

            print('Done.')
            return

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
