# Imports
import cv2
import json
import os
import torch
import torchvision
import test_algorithms

import numpy as np
import torch.onnx as onnx
import torchvision.models as models
import transforms as T

from engine import train_one_epoch, evaluate
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.ssd import SSDFeatureExtractorVGG


# Global variables used for printing results
MIN_CONFIDENCE = 0.8
TRAIN_TEST_SPLIT = 0.8
IMAGE_DIR = "_out"
GROUND_TRUTH_DIR = "GroundTruthRGB"
LABEL_DIR = "Objects"

CLASSES = {1: "Person", 2: "Bicycle", 3: "Vehicle", 4: "Motorbike", 6: "Bus", 7: "Train", 8: "Truck", 9: "Boat", 10: "Traffic LIght",
           12: "Street Sign", 13: "Stop Sign", 14: "Parking Meter", 17: "Cat", 18: "Dog", 19: "Horse", 20: "Sheep", 21: "Cow", 25: "Other"}
CARLA_TO_COCO_ID_CONVERSION = {4: 1, 10: 3, 12: 12}
COLORS = np.random.uniform(0, 255, size=(25, 3))


# Helper Functions
# Make model be ready to train on custom CARLA vehicle dataset
def get_model_instance_segmentation():
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.ssd300_vgg16(pretrained=True)

    # TODO: Update to ensure new labels match existing

    return model


# Helper function for data augmentation and transformation
# Simplifies training and evaluation detection models
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


# Load the NN
def load_nn_model(idx):
    model = torch.load(os.path.join(os.getcwd(), "models", 'model' + str(idx) + '.pt'))
    model = model.cuda()
    print("Successfully loaded model!")
    return model


# Predict using a trained model
def predict(model, data):
    pred = model(data)
    return pred


# Visualizes predictions
# Source: https://www.pyimagesearch.com/2021/08/02/pytorch-object-detection-with-pre-trained-networks/
def visualize_predictions(orig, detections):
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
            # display the prediction to our terminal
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            print("[INFO] {}".format(label))
            # draw the bounding box and label on the image
            cv2.rectangle(orig, (startX, startY), (endX, endY),
                COLORS[idx-1], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(orig, label, (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx-1], 2)

    # show the output image
    cv2.imshow("Output", orig)
    cv2.waitKey(0)


# Kicks off prediction on the ground truth images
def predict_ground_truth_results(model):
    transform = T.ToTensor()
    imgs = list(sorted(os.listdir(os.path.join(os.getcwd(), GROUND_TRUTH_DIR))))
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    for path in imgs:
        img_path = os.path.join(os.getcwd(), GROUND_TRUTH_DIR, path)
        orig = cv2.imread(img_path)
        orig = orig[0: 390, 0: 799]
        img = Image.open(img_path).convert("RGB")
        img = img.crop((0, 0, 799, 390))
        input = transform(img)[0]
        img = input.unsqueeze_(0)
        model.eval()
        pred = model(input.cuda())[0]
        visualize_predictions(orig, pred)


# Main Method
def main():
    # Load model and predict (use pre-trained OR loaded model)
    # model = get_model_instance_segmentation()
    model = load_nn_model(6)
    predict_ground_truth_results(model)


if __name__ == "__main__":
    main()
