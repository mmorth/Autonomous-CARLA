# Imports
import cv2
import json
import os
import numpy as np


# Constants and global variables
CLASSES = [4, 10, 12] # Classes according to CARLA's semantic segmentation sensor: https://carla.readthedocs.io/en/stable/cameras_and_sensors/

# Loop through each captured episode images
episodes = list(sorted(os.listdir(os.path.join(os.getcwd(), "_out"))))
imgs_labeled = 0
for episode in episodes:
    print("Processing {}...".format(episode))

    # Locate the sematic segmentated images in the filesystem
    imgs = list(sorted(os.listdir(os.path.join(os.getcwd(), "_out", episode, "CameraSemanticSegmentation"))))
    num_imgs = 0
    for path in imgs:
        # Read and convert image to grayscale
        img_path = os.path.join(os.getcwd(), "_out", episode, "CameraSemanticSegmentation", path)
        img = cv2.imread(img_path)

        # Create variable to store JSON data
        data = {}
        data['objects'] = []

        # Process each desired clss (1=buildings, 4=pedestrains, 6=road lines, 10=vehicles, 12=traffic signs)
        for tag in CLASSES:
            # Bitwise mask to isolate only the current tag in the image
            lower_tag_range = np.array([0,0,tag], dtype = "uint16")
            upper_tag_range = np.array([0,0,tag], dtype = "uint16")
            tag_mask = cv2.inRange(img, lower_tag_range, upper_tag_range)

            # TODO: Determine whether to save the masks as well for training a masked CNN

            # TODO: Determine whether to include these morphological operations in the pre-processing
            # se = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
            # closing = cv2.morphologyEx(tag_mask, cv2.MORPH_CLOSE, se)

            # cv2.imshow("Closing", closing)
            # cv2.waitKey(0)

            # Use OpenCV to find the bounding box location of the object
            contours, hierarchy = cv2.findContours(tag_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                # Handle overlapping classes (e.g. 2 vehicles with overlapping ROIs that make it look like 1 ROI) (could manually label in this case)
                    # This would be an issue since both would be in the same class, so you can classify the entire blob as one class even if they are separate objects
                    # The only area this may be an issue is when doing visual odometry or object tracking, but you could use the previous motion model to track its position
                # TODO: Handle situations when there is a gap (e.g. between arm and body) that creates a separate detection (maybe require certain area for detection?)
                    # This could be handled by using morphological operations to remove or reduce these, but you would need to be careful not to join two separate objects as a result
                    # This could also be handled by having a minimum area requirement for the detected object annotation
                    # Additionally, storing all detected boxes and using Non-Maximum Suppression to remove duplicate detections could work
                # TODO: Remove duplicate detections for the same object or when one object gets detected as two (for split masks) (maybe morphological operations?)
                    # This could be resolved by the area requirement as this would only likely occur for objects far away.
                    # Additionally, morphological operations could be done to reduce these affects
                if cv2.contourArea(cnt) > 200 and cv2.contourArea(cnt) < 100000:
                    x,y,w,h = cv2.boundingRect(cnt)
                    data['objects'].append({
                        "label": tag,
                        "x0": x,
                        "y0": y,
                        "x1": x+w,
                        "y1": y+h
                    })
                    # x,y,w,h = cv2.boundingRect(cnt)
                    # cv2.rectangle(tag_mask,(x,y),(x+w,y+h),(255,255,255),1)
                    # cv2.imshow("Box", tag_mask)
                    # cv2.waitKey(0)

        num_imgs += 1

        if num_imgs % 100 == 0:
            print("Processed {} images".format(num_imgs))

        # Create an output file in COCO format annotating the bounding box and class location for each object
        json_file_path = os.path.join(os.getcwd(), "Objects", episode)

        if not os.path.exists(json_file_path):
            os.makedirs(json_file_path)

        with open(os.path.join(json_file_path, path[:-4] + ".json"), 'w') as outfile:
            json.dump(data, outfile)
        
    imgs_labeled += num_imgs
    
print("Autolabeling Complete! Auto-labeled {} images.".format(imgs_labeled))
