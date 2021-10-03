# Imports
import cv2
import json
import os
import torch
import torchvision
import utils

import numpy as np
import torch.onnx as onnx
import torchvision.models as models
import transforms as T

from engine import train_one_epoch, evaluate
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.ssd import SSDFeatureExtractorVGG

'''
This Python script is used to train a Neural Network to detect objects from the CARLA simulation environment.
Inputs: 
    1. RGB images captured from the vehicle in the CARLA environment
    2. A JSON file in the format from the auto_labeler script, with object class label and bounding box coordinate location
Outputs:
    1. A trained .pt model from the input data
The training utilizes Transfer Learning from a pre-trained COCO VGG16 network.
Make sure to modify the global variables below to fit your desired input.
'''

# Global variables used for printing results
TRAIN_TEST_SPLIT = 0.8
NUM_TRAINING_EPOCHS = 50
IMAGE_DIR = "_out"
GROUND_TRUTH_DIR = "GroundTruthRGB"
LABEL_DIR = "Objects"
# Eventually, these are the COCO class labels we want to detect
CLASSES = {1: "Person", 2: "Bicycle", 3: "Vehicle", 4: "Motorbike", 6: "Bus", 7: "Train", 8: "Truck", 9: "Boat", 10: "Traffic LIght",
           12: "Street Sign", 13: "Stop Sign", 14: "Parking Meter", 17: "Cat", 18: "Dog", 19: "Horse", 20: "Sheep", 21: "Cow", 25: "Other"}
# Conversion table CARLA to COCO classes
CARLA_TO_COCO_ID_CONVERSION = {4: 1, 10: 3, 12: 12}
NUM_CLASSES_TO_DETECT = 3
COLORS = np.random.uniform(0, 255, size=(25, NUM_CLASSES_TO_DETECT))


# Create Dataset: Converts labeled images into PyTorch label format
# Source (modified): https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html#
class CarlaVehicleDataset(torch.utils.data.Dataset):

    # Initializes the dataset by defining the transform and finding and storing the names of the RGB images and labels/bounding box JSON files
    def __init__(self, root, transforms):
        print("Creating and pre-processing the dataset...")
        self.root = root
        # store the data pre-processing transform
        self.transforms = transforms
        # load all image and JSON files, sorting them to ensure that they are aligned
        episodes = list(sorted(os.listdir(os.path.join(root, IMAGE_DIR))))
        self.imgs = []
        for episode in episodes:
            self.imgs.extend(sorted(os.listdir(os.path.join(root, IMAGE_DIR, episode, "CameraRGB"))))

        self.labels = []
        for episode in episodes:
            self.labels.extend(sorted(os.listdir(os.path.join(root, LABEL_DIR, episode))))


    # Stores the relevant information from image labeling for use in PyTorch
    def __getitem__(self, idx):
        # load images and JSON files
        episode_num = self.imgs[idx][0:4]
        img_path = os.path.join(self.root, IMAGE_DIR, "episode_" + episode_num, "CameraRGB", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        label_path = os.path.join(self.root, LABEL_DIR, "episode_" + episode_num, self.labels[idx])
        label_file = open(label_path,)
        label = json.load(label_file)

        # get bounding box coordinates for each label
        num_objs = len(label['objects'])
        boxes = []
        labels = []
        for i in range(num_objs):
            xmin = label['objects'][i]['x0']
            xmax = label['objects'][i]['x1']-1
            ymin = label['objects'][i]['y0']
            ymax = label['objects'][i]['y1']-1
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(CARLA_TO_COCO_ID_CONVERSION[label['objects'][i]['label']])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # store in PyTorch format
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


    def __len__(self):
        return len(self.imgs)


# Carla Object Detection NN: Trains and saves an object detection model for vehicles, pedestrains, and traffic signs in the CARLA simulation environment
class CarlaObjectDetectionNN():

    # Make model be ready to train on custom CARLA vehicle dataset
    def get_model_instance_segmentation(self):
        # load an VGG16 model pre-trained on COCO for CARLA transfer learning
        model = torchvision.models.detection.ssd300_vgg16(pretrained=True)

        return model


    # Helper function for data augmentation and transformation
    # Simplifies training and evaluation detection models by normalizing image inputs to training
    def get_transform(self, train):
        transforms = []
        transforms.append(T.ToTensor())
        if train:
            transforms.append(T.RandomHorizontalFlip(0.5))
        return T.Compose(transforms)


    # Train and Validate the NN
    def train_and_validate_nn(self, model=None):
        # train on the GPU or on the CPU, if a GPU is not available
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # obtain dataset of training and validation images and JSON labels/bounding boxes
        dataset = CarlaVehicleDataset(os.getcwd(), self.get_transform(train=True))
        dataset_test = CarlaVehicleDataset(os.getcwd(), self.get_transform(train=False))

        # randomly split the dataset in train and test set
        indices = torch.randperm(len(dataset)).tolist()
        test_begin = int(len(dataset) * TRAIN_TEST_SPLIT)
        train_end = int(len(dataset) - test_begin)

        dataset = torch.utils.data.Subset(dataset, indices[:-train_end])
        dataset_test = torch.utils.data.Subset(dataset_test, indices[test_begin:])

        print("Train Set Size: {}".format(len(dataset)))
        print("Test Set Size: {}".format(len(dataset_test)))

        # TODO: Tune hyperparameters using RayTune

        # define training and validation data loaders
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=16, shuffle=True, num_workers=4,
            collate_fn=utils.collate_fn)

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=8, shuffle=False, num_workers=4,
            collate_fn=utils.collate_fn)

        # get the model using our helper function
        if model == None:
            model = self.get_model_instance_segmentation()

        # move model to the right device
        model.to(device)

        # construct an optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=0.001)
        # and a learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=3,
                                                    gamma=0.1)

        for epoch in range(NUM_TRAINING_EPOCHS):
            # train for one epoch
            train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
            # update the learning rate
            lr_scheduler.step()
            # evaluate on the test dataset
            evaluate(model, data_loader_test, device=device)

            print("Saving model as model{}.pt...".format(epoch))
            self.save_nn_model(model, epoch)


        print("Model Training Complete!")
        return model


    # Save the NN
    def save_nn_model(self, model, idx):
        torch.save(model, os.path.join(os.getcwd(), "models", 'model' + str(idx) + '.pt'))
        print("Model successfully saved!")


    # Load the NN
    def load_nn_model(self, idx):
        model = torch.load(os.path.join(os.getcwd(), "models", 'model' + str(idx) + '.pt'))
        model = model.cuda()
        print("Successfully loaded model!")
        return model


# Main Method
def main():
    # Train and save model on new data
    carla_object_detection_nn = CarlaObjectDetectionNN()
    # model = carla_object_detection_nn.load_nn_model(0)
    model = carla_object_detection_nn.train_and_validate_nn()
    # carla_object_detection_nn.save_nn_model(model, 1)


if __name__ == "__main__":
    main()
