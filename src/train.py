import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from tqdm import tqdm

import utils.config as cfg
from datasets.landmark_dataset import LandmarkDataset
from models.detection_model import BboxDetectionModel
from utils.models_utils import ModelUtils, TrainTransforms


# MEAN, STD and RESIZE_SIZE declaration
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
RESIZE_SIZE = (400, 300)
BBOX_WEIGHT = 1.0
LABEL_WEIGHT = 1.0

INITIAL_LR = 1e-4
NUM_EPOCHS = 20
BATCH_SIZE = 64


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False  # to speed up loading data on CPU to training it on GPU
# see https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723

tf_list = TrainTransforms(mean=MEAN, std=STD, resize_size=RESIZE_SIZE)

dataset = LandmarkDataset(
    root="./" + cfg.LANDMARK_DATASET_PATH, runing_mode="train", device="cpu", transforms_list=tf_list
)
train_dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, num_workers=os.cpu_count(), pin_memory=PIN_MEMORY)


resnet_utils = ModelUtils()
resnet_model = resnet_utils.build_model(model=resnet50, pretrained=True, gradation=False)

# creating object dedector model
detector_model = BboxDetectionModel(resnet_model, 3)

if torch.cuda.is_available():
    detector_model.cuda()

# loss funcfion declaration
classificator_loss_func = nn.CrossEntropyLoss()
bbox_loss_func = nn.MSELoss()
# optimizer initialization
optimizer = torch.optim.Adam(params=detector_model.parameters(), lr=INITIAL_LR)

H = {"total_train_loss": [], "total_val_loss": [], "train_class_acc": [], "val_class_acc": []}


print(f"{cfg.TERMINAL_INFO} start training loop...")
start_time = time.time()

for epoch in tqdm(range(NUM_EPOCHS)):
    detector_model.train()

    # initialization losses
    total_train_loss = 0
    total_valid_loss = 0

    # initialization predictions num
    train_correct = 0
    valid_correct = 0
    i = 0
    for (image, label, bbox) in train_dataloader:

        image = image.to(DEVICE)
        label = label.to(DEVICE)
        bbox = bbox.to(DEVICE)

        predictions = detector_model(image)
        bbox_loss = bbox_loss_func(predictions[0], bbox.float())
        classificator_loss = classificator_loss_func(predictions[1], label)

        total_loss = bbox_loss + classificator_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        total_train_loss += total_loss
        train_correct += (predictions[1].argmax(1) == label).type(torch.float).sum().item()

        i += 1
        if i % 10 == 0:
            print(i, end="\r")

    print(end="\x1b[2K")
    # TODO valid loop
