# Imports
import cv2
import json
import os
import numpy as np

'''
This Python script automatically labels vehicles, pedestrains, and traffic signs using the following algorithm:
1. Threshold the semantic segmentation image into images containing just a single class
    a. All class pixels are set to 255 while all non-class pixels are set to 0
2. Perform contour detection on each object from the thresholded class image
3. For each detected contour:
    a. Find and store the bounding box coordinates of the contour if it's area is sufficiently large
This process will label every object with its location (using contour analysis) and class (from the semantic segmentation image).
Images should be stored in a directory named _out/episode_#####
'''

# Constants and global variables
CLASSES = [4, 10, 12] # Classes according to CARLA's semantic segmentation sensor: https://carla.readthedocs.io/en/stable/cameras_and_sensors/

# Loop through each captured episode images
episodes = list(sorted(os.listdir(os.path.join(os.getcwd(), "_out"))))
imgs_labeled = 0
for episode in episodes:
    print("Processing {}...".format(episode))

    # Locate and crete a list of the sematic segmentated images in the filesystem
    imgs = list(sorted(os.listdir(os.path.join(os.getcwd(), "_out", episode, "CameraSemanticSegmentation"))))
    num_imgs = 0
    for path in imgs:
        # Read and store image
        img_path = os.path.join(os.getcwd(), "_out", episode, "CameraSemanticSegmentation", path)
        img = cv2.imread(img_path)

        # Create variable to store JSON data (class label and object location)
        data = {}
        data['objects'] = []

        # Process each desired clss (4=pedestrains, 10=vehicles, 12=traffic signs)
        for tag in CLASSES:
            # Bitwise mask to isolate only the current tag in the image
            lower_tag_range = np.array([0,0,tag], dtype = "uint16")
            upper_tag_range = np.array([0,0,tag], dtype = "uint16")
            tag_mask = cv2.inRange(img, lower_tag_range, upper_tag_range)

            # Use OpenCV to find the bounding box location of the object
            contours, hierarchy = cv2.findContours(tag_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                # Ensure minimum contour area and maximum contour area (to not detect front of vehicle)
                if cv2.contourArea(cnt) > 200 and cv2.contourArea(cnt) < 100000:
                    x,y,w,h = cv2.boundingRect(cnt)
                    data['objects'].append({
                        "label": tag,
                        "x0": x,
                        "y0": y,
                        "x1": x+w,
                        "y1": y+h
                    })

        num_imgs += 1

        # Debugging progress
        if num_imgs % 100 == 0:
            print("Labeled {} images".format(num_imgs))

        # Remove images with no object annotations
        if len(data['objects']) == 0:
            os.remove(img_path)
            rgb_path = os.path.join(os.getcwd(), "_out", episode, "CameraRGB", path)
            os.remove(rgb_path)
            print("[NOTE]: Deleted image " + str(path) + " with no labels")
        else:
            # Create an output file in COCO format annotating the bounding box and class label for each object
            json_file_path = os.path.join(os.getcwd(), "Objects", episode)

            if not os.path.exists(json_file_path):
                os.makedirs(json_file_path)

            with open(os.path.join(json_file_path, path[:-4] + ".json"), 'w') as outfile:
                json.dump(data, outfile)
        
    imgs_labeled += num_imgs
    
print("Autolabeling Complete! Auto-labeled {} images.".format(imgs_labeled))
