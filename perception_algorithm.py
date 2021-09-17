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


# Visualizes predictions
# Source: https://www.pyimagesearch.com/2021/08/02/pytorch-object-detection-with-pre-trained-networks/
def visualize_predictions(orig, depth_img, detections):
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
            
            # compute depth information from the depth map
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            slice = depth_img[startY: endY, startX: endX]
            depth = np.average(slice)
            distance = 1000 * depth
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

            if args.lidar:
                lidar = Lidar('Lidar32')
                lidar.set_position(0, 0, 2.50)
                lidar.set_rotation(0, 0, 0)
                lidar.set(
                    Channels=32,
                    Range=50,
                    PointsPerSecond=100000,
                    RotationFrequency=10,
                    UpperFovLimit=10,
                    LowerFovLimit=-30)
                settings.add_sensor(lidar)

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
                    visualize_predictions(orig, depth_img, pred)
                elif name == 'CameraDepth':
                    # Obtain and display the depth image
                    depth_img = depth_to_array(measurement)

                    # TODO: Compute the object's depth
                    # in_meters = 1000 * depth_img[x,y]

                    cv2.imshow("CameraDepth", depth_img)
                    cv2.waitKey(1)

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
