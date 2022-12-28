import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from tqdm import tqdm

import utils.config as cfg
from datasets.landmark_dataset import LandmarkDataset
from datasets.lm_hdf5_dataset import LandmarkHDF5Dataset
from models.detection_model import BboxDetectionModel
from utils.models_utils import ModelUtils, TrainTransforms


# MEAN, STD and RESIZE_SIZE declaration
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
RESIZE_SIZE = (200, 200)
BBOX_WEIGHT = 1.0
LABEL_WEIGHT = 1.0

INITIAL_LR = 1e-4
NUM_EPOCHS = 20
BATCH_SIZE = 32

AS_TENSOR = False
DEBUG_MODE = False

NUM_WORKERS = 8
# NUM_WORKERS = os.cpu_count()


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False  # to speed up loading data on CPU to training it on GPU
# see https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723

tf_list = TrainTransforms(mean=MEAN, std=STD, as_hdf5=True)

dataset = LandmarkHDF5Dataset(
    root=cfg.HDF5_DIR_PATH,
    running_mode="train",
    transforms_list=tf_list,
    measure_time=True,
)

# tf_list = TrainTransforms(mean=MEAN, std=STD, resize_size=RESIZE_SIZE, as_tensor=AS_TENSOR)

# dataset = LandmarkDataset(
#     root="./" + cfg.LANDMARK_DATASET_PATH,
#     runing_mode="train",
#     device="cpu",
#     transforms_list=tf_list,
#     other_dir_name=None,
#     load_as_tensors=AS_TENSOR,
#     measure_time=False,
# )

train_dataloader = DataLoader(
    dataset=dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
)


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
    loading_time_start = time.time()

    for (image, label, bbox) in train_dataloader:

        it_time_start = time.time()

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
        it_time_end = time.time()
        if i % 10 == 0 and DEBUG_MODE:
            measured_time = it_time_end - it_time_start
            loading_time = it_time_end - loading_time_start
            print(
                f"iterations: {i}\t time in it: {measured_time:.2f}\t loading time per it: {(loading_time/i):.2f}",
                end="\r",
            )
    if DEBUG_MODE:
        print(end="\x1b[2K")

    # TODO valid loop
