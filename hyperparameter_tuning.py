# Imports
import cv2
import json
import os
from numpy.random import gamma
import torch
import torchvision
import utils

import numpy as np
import torch.onnx as onnx
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
import transforms as T

from engine import train_one_epoch, evaluate
from functools import partial
from PIL import Image
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import random_split
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.ssd import SSDFeatureExtractorVGG


# Constants
MIN_CONFIDENCE = 0.2
TRAIN_TEST_SPLIT = 0.8
IMAGE_DIR = "CameraRGB"
GROUND_TRUTH_DIR = "GroundTruthRGB"
LABEL_DIR = "CameraRGB_Objs"


# Create Dataset: Converts labeled images into PyTorch label format
# Source: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html#
class CarlaVehicleDataset(torch.utils.data.Dataset):

    def __init__(self, root, transforms):
        self.root = root
        # store the data pre-processing transform
        self.transforms = transforms
        # load all image files, sorting them to ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, IMAGE_DIR))))
        self.labels = list(sorted(os.listdir(os.path.join(root, LABEL_DIR))))


    # stores the relevant information from image labeling
    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, IMAGE_DIR, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        label_path = os.path.join(self.root, LABEL_DIR, self.labels[idx])
        label_file = open(label_path,)
        label = json.load(label_file)

        # get bounding box coordinates for each label
        num_objs = len(label[0]['annotations'])
        boxes = []
        for i in range(num_objs):
            xmin = label[0]['annotations'][i]['coordinates']['x'] - label[0]['annotations'][i]['coordinates']['width'] / 2
            xmax = label[0]['annotations'][i]['coordinates']['x'] + label[0]['annotations'][i]['coordinates']['width'] / 2
            ymin = label[0]['annotations'][i]['coordinates']['y'] - label[0]['annotations'][i]['coordinates']['height'] / 2
            ymax = label[0]['annotations'][i]['coordinates']['y'] + label[0]['annotations'][i]['coordinates']['height'] / 2
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

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


# Helper Functions
# Make model be ready to train on custom CARLA vehicle dataset
def get_model_instance_segmentation(num_classes=1):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.ssd300_vgg16(pretrained=True)

    # TODO: Update to use num_classes and ensure image annotations match trained model classes

    return model


# Helper function for data augmentation and transformation
# Simplifies training and evaluation detection models
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


###############################################################################################################
###############################################################################################################
###################################### Hyperparameter Tuning ##################################################
###############################################################################################################
###############################################################################################################


# Wrap the data loaders in their own function and pass global data directory
# This allows sharing of a data directory between different trials
def load_data(data_dir="./data"):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = CarlaVehicleDataset(os.getcwd(), get_transform(train=True))
    dataset_test = CarlaVehicleDataset(os.getcwd(), get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    test_begin = len(dataset) * TRAIN_TEST_SPLIT
    train_end = len(dataset) - test_begin
    dataset = torch.utils.data.Subset(dataset, indices[:-train_end])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[test_begin:])

    return dataset, dataset_test


# Training with Ray Tune hyperparameter tuning
def train_vgg(config, checkpoint_dir=None, data_dir=None):
    net = get_model_instance_segmentation(17)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config["learning_rate"], momentum=config["momentum"], weight_decay=config["weight_decay"])

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    trainset, testset = load_data(data_dir)

    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs])

    trainloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=config["num_workers"])
    valloader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=config["num_workers"])

    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((net.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps), accuracy=correct / total)
    print("Finished Training")


# Test accuracy of current model
def test_accuracy(net, device="cpu"):
    trainset, testset = load_data()

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


# Main method used for RayTune Hyperparameter tuning
def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    data_dir = os.path.abspath("./data")
    load_data(data_dir)
    config = {
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "num_workers": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "momentum": tune.uniform(0.0, 1.0),
        "weight_decay": tune.uniform(0.0, 1.0),
        "gamma": tune.uniform(0.0, 1.0),
        "step_size": tune.sample_from(lambda _: np.random.randint(1, 9)),
        "batch_size": tune.choice([2, 4, 8, 16])
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    result = tune.run(
        partial(train_vgg, data_dir=data_dir),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        step_size=config["step_size"],
        gamma=config["gamma"])

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    best_trained_model = get_model_instance_segmentation(17)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    print("Best Config Hyperparameters:")
    print("batch_size: {}".format(best_trial["batch_size"]))
    print("num_workers: {}".format(best_trial["num_workers"]))
    print("learning_rate: {}".format(best_trial["learning_rate"]))
    print("momentum: {}".format(best_trial["momentum"]))
    print("weight_decay: {}".format(best_trial["weight_decay"]))
    print("step_size: {}".format(best_trial["step_size"]))
    print("gamma: {}".format(best_trial["gamma"]))

    test_acc = test_accuracy(best_trained_model, device)
    print("Best trial test set accuracy: {}".format(test_acc))


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=10, max_num_epochs=10, gpus_per_trial=0)
