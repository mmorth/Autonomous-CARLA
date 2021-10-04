'''
This function calls the relevant perception, controls, state estimation/localization, and mapping functions to execute the CARLA autonomous vehicle stack.
'''

# Support legacy print functions
from __future__ import print_function

# Library and utility imports
import argparse
import cv2
import keyboard
import logging
import math
import os
import random
import time
import torch

import numpy as np
import pytorch_utils.transforms as T

from numpy.testing._private.utils import measure
from object_detection_nn import CLASSES, COLORS

# CARLA API Imports
from carla.client import make_carla_client
from carla.image_converter import to_bgra_array, depth_to_array
from carla.sensor import Camera, Lidar
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line
from carla.client import make_carla_client, VehicleControl

# Autonomous-CARLA Stack Imports
import perception_stack


# A class that represents real-time autonomous CARLA execution
class AutonomousCARLA():

    # Initialize local variables
    def __init__(self, model_num, enable_autopilot=False):
        self.reverse_on = False
        self.enable_autopilot = enable_autopilot
        self.model_num = model_num


    # Receives the keyboard inputs from the user and constructs a VehicleControl object based on inputs
    def get_keyboard_control(self):
        control = VehicleControl()
        if keyboard.is_pressed('a'):
            control.steer = -1.0
        if keyboard.is_pressed('d'):
            control.steer = 1.0
        if keyboard.is_pressed('w'):
            control.throttle = 1.0
        if keyboard.is_pressed('s'):
            control.brake = 1.0
        if keyboard.is_pressed('q'):
            reverse_on = not self.reverse_on
        if keyboard.is_pressed('p'):
            enable_autopilot = not self.enable_autopilot
        control.reverse = self.reverse_on
        return control


    def configure_settings_and_sensors(self, args, client):
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

        # Load these settings into the server
        scene = client.load_settings(settings)

        # Choose one player start at random.
        number_of_player_starts = len(scene.player_start_spots)
        player_start = random.randint(0, max(0, number_of_player_starts - 1))

        # Notify the server that we want to start the episode at the
        # player_start index. This function blocks until the server is ready
        # to start the episode.
        print('Starting new episode at %r...' % scene.map_name)
        client.start_episode(player_start)


    # Runs the CARLA client utilizing the Autonomous-CARLA stack
    def run_carla_client(self, args):
        camera_fov = 0

        with make_carla_client(args.host, args.port) as client:
            print('CarlaClient connected')
            
            # Load object detection model
            model = torch.load(os.path.join(os.getcwd(), "models", 'model' + str(self.model_num) + '.pt'))
            model = model.cuda()
            print("Successfully loaded model!")

            self.configure_settings_and_sensors(args, client)

            # Iterate every frame in the simulation
            frame = 0
            depth_img = None
            ss_img = None
            cur_img = None
            prev_img = None
            while True:
                frame += 1

                # Read the data produced by the server this frame.
                measurements, sensor_data = client.read_data()

                # Print some of the measurements.
                self.print_measurements(measurements)

                # Save the images to disk if requested.
                for name, measurement in sensor_data.items():
                    if name == 'CameraRGB':
                        # Obtain and detect objects on the RGB image
                        img = to_bgra_array(measurement)
                        cur_img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                        # TODO: Compute and display object positions, depth, and lane lines

                        prev_img = cur_img
                    elif name == 'CameraDepth':
                        # Obtain and save the depth measurements for distance visualization
                        depth_img = depth_to_array(measurement)
                    elif name == 'CameraSemanticSegmentation':
                        # Obtain and save the semantic segmentation measurements for lane visualization
                        ss_img = to_bgra_array(measurement)

                # Send the position to the client
                control = self.get_keyboard_control()

                if self.enable_autopilot:
                    client.send_control(measurements.player_measurements.autopilot_control)
                else:
                    client.send_control(control)


    def print_measurements(self, measurements):
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


# Main method execution
def main():
    # Create Autonomous-CARLA reference
    autonomous_carla = AutonomousCARLA(0)

    # Parse configurable arguments
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
        '-q', '--quality-level',
        choices=['Low', 'Epic'],
        type=lambda s: s.title(),
        default='Low',
        help='graphics quality level, a lower level makes the simulation run considerably faster.')
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
            autonomous_carla.run_carla_client(args)

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
