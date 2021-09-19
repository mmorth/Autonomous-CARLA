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


# Global variables used for printing results
MIN_CONFIDENCE = 0.5
TRAIN_TEST_SPLIT = 0.8
IMAGE_DIR = "_out"
GROUND_TRUTH_DIR = "GroundTruthRGB"
LABEL_DIR = "Objects"

CLASSES = {1: "Person", 2: "Bicycle", 3: "Vehicle", 4: "Motorbike", 6: "Bus", 7: "Train", 8: "Truck", 9: "Boat", 10: "Traffic LIght",
           12: "Street Sign", 13: "Stop Sign", 14: "Parking Meter", 17: "Cat", 18: "Dog", 19: "Horse", 20: "Sheep", 21: "Cow", 25: "Other"}
CARLA_TO_COCO_ID_CONVERSION = {4: 1, 10: 3, 12: 12}
COLORS = np.random.uniform(0, 255, size=(25, 3))


# Create Dataset: Converts labeled images into PyTorch label format
# Source (modified): https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html#
class CarlaVehicleDataset(torch.utils.data.Dataset):

    def __init__(self, root, transforms):
        print("Creating and pre-processing the dataset...")
        self.root = root
        # store the data pre-processing transform
        self.transforms = transforms
        # load all image files, sorting them to ensure that they are aligned
        episodes = list(sorted(os.listdir(os.path.join(root, IMAGE_DIR))))
        self.imgs = []
        for episode in episodes:
            self.imgs.extend(sorted(os.listdir(os.path.join(root, IMAGE_DIR, episode, "CameraRGB"))))

        self.labels = []
        for episode in episodes:
            self.labels.extend(sorted(os.listdir(os.path.join(root, LABEL_DIR, episode))))

        # # Pre-process filter invalid boxes
        # self.imgs = []
        # self.labels = []
        # for idx in range(len(imgs)):
        #     episode_num = imgs[idx][0:4]
        #     label_path = os.path.join(self.root, LABEL_DIR, "episode_" + episode_num, labels[idx])
        #     label_file = open(label_path,)
        #     label = json.load(label_file)

        #     # Only add annotations with valid bounding boxes
        #     num_objs = len(label['objects'])
        #     all_boxes_valid = True
        #     for i in range(num_objs):
        #         # print("=========================================")
        #         # print("x0 = {}".format(label['objects'][i]['x0']))
        #         # print("x1 = {}".format(label['objects'][i]['x1']))
        #         # print("y0 = {}".format(label['objects'][i]['y0']))
        #         # print("y1 = {}".format(label['objects'][i]['y1']))
        #         # print("=========================================")
        #         if label['objects'][i]['x0'] >= label['objects'][i]['x1'] \
        #             or label['objects'][i]['y0'] >= label['objects'][i]['y1'] \
        #             or label['objects'][i]['x0'] < 0 \
        #             or label['objects'][i]['x1']-1 >= 799 \
        #             or label['objects'][i]['y0'] < 0 \
        #             or label['objects'][i]['y1']-1 >= 390:
        #             all_boxes_valid = False
        #             # print("INVALID BOXES DETECTED!!!!!")
            
        #     if all_boxes_valid:
        #         self.imgs.append(imgs[idx])
        #         self.labels.append(labels[idx])

        #     if idx % 1000 == 0:
        #         print("Pre-processed {} images".format(idx))


    # stores the relevant information from image labeling
    def __getitem__(self, idx):
        # load images and masks
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


# Train and Validate the NN
def train_and_validate_nn(model=None):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 3
    # use our dataset and defined transformations
    dataset = CarlaVehicleDataset(os.getcwd(), get_transform(train=True))
    dataset_test = CarlaVehicleDataset(os.getcwd(), get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    test_begin = int(len(dataset) * TRAIN_TEST_SPLIT)
    train_end = int(len(dataset) - test_begin)

    dataset = torch.utils.data.Subset(dataset, indices[:-train_end])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[test_begin:])

    print("Train Set Size: {}".format(len(dataset)))
    print("Test Set Size: {}".format(len(dataset_test)))

    # TODO: Tune hyperparameters

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    if model == None:
        model = get_model_instance_segmentation()

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0001,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 50

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

        print("Saving model as model{}.pt...".format(epoch))
        save_nn_model(model, epoch)


    print("Model Training Complete!")
    return model


# Save the NN
def save_nn_model(model, idx):
    torch.save(model, os.path.join(os.getcwd(), "models", 'model' + str(idx) + '.pt'))
    print("Model successfully saved!")


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
        img = Image.open(img_path).convert("RGB")
        input = transform(img)[0]
        img = input.unsqueeze_(0)
        model.eval()
        pred = model(input.cuda())[0]
        visualize_predictions(orig, pred)


# Main Method
def main():
    # Train and save model on new data
    # model = load_nn_model(0)
    # model = train_and_validate_nn()
    # save_nn_model(model, 1)

    # # Load model and predict (use pre-trained OR loaded model)
    # model = get_model_instance_segmentation()
    model = load_nn_model(10)
    predict_ground_truth_results(model)


if __name__ == "__main__":
    main()
