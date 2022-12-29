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
BATCH_SIZE = 64

AS_TENSOR = False
DEBUG_MODE = False

NUM_WORKERS = 8
# NUM_WORKERS = os.cpu_count()


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False  # to speed up loading data on CPU to training it on GPU
# see https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723

tf_list = TrainTransforms(mean=MEAN, std=STD, as_hdf5=True)

train_dataset = LandmarkHDF5Dataset(
    root=cfg.HDF5_DIR_PATH,
    running_mode="train",
    transforms_list=tf_list,
    measure_time=False,
)
val_dataset = LandmarkHDF5Dataset(
    root=cfg.HDF5_DIR_PATH,
    running_mode="val",
    transforms_list=tf_list,
    measure_time=False,
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
    dataset=train_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
)

val_dataloader = DataLoader(
    dataset=val_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
)

train_steps = len(train_dataset) // BATCH_SIZE
val_steps = len(val_dataset) // BATCH_SIZE

print(cfg.TERMINAL_INFO, f"Train photos:{len(train_dataset)}\tVal photos: {len(val_dataset)}")
resnet_utils = ModelUtils()
resnet_model = resnet_utils.build_model(model=resnet50, pretrained=True, gradation=False)

# creating object dedector model
detector_model = BboxDetectionModel(resnet_model, 3)

if torch.cuda.is_available():
    detector_model.cuda()

# loss funcfion declaration (ctiterion)
classificator_loss_func = nn.CrossEntropyLoss()
bbox_loss_func = nn.MSELoss()
# optimizer initialization
optimizer = torch.optim.Adam(params=detector_model.parameters(), lr=INITIAL_LR)

progress = {"total_train_loss": [], "total_val_loss": [], "train_class_acc": [], "val_class_acc": []}


print(f"{cfg.TERMINAL_INFO} start training loop...")
start_time = time.time()

for epoch in tqdm(range(NUM_EPOCHS)):
    detector_model.train()

    # initialization losses
    total_train_loss = 0
    total_val_loss = 0

    # initialization predictions num
    correct_train = 0
    correct_val = 0
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

        # total_loss = bbox_loss + classificator_loss
        total_loss = classificator_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        total_train_loss += total_loss
        correct_train += (predictions[1].argmax(1) == label).type(torch.float).sum().item()

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

    with torch.no_grad():
        detector_model.eval()

        for (image, label, bbox) in val_dataloader:

            image = image.to(DEVICE)
            label = label.to(DEVICE)
            bbox = bbox.to(DEVICE)

            predictions = detector_model(image)

            bbox_loss = bbox_loss_func(predictions[0], bbox.float())
            classificator_loss = classificator_loss_func(predictions[1], label)

            # total_loss = bbox_loss + classificator_loss
            total_loss = classificator_loss

            total_val_loss += total_loss

            correct_val += (predictions[1].argmax(1) == label).type(torch.float).sum().item()

    avg_train_loss = total_train_loss / train_steps
    avg_val_loss = total_val_loss / val_steps

    correct_train = correct_train / len(train_dataset)
    correct_val = correct_val / len(val_dataset)

    # update training history
    progress["total_train_loss"].append(avg_train_loss.cpu().detach().numpy())
    progress["total_val_loss"].append(avg_val_loss.cpu().detach().numpy())
    progress["train_class_acc"].append(correct_train)
    progress["val_class_acc"].append(correct_val)

    #!MODEL TRAINGING INFO

    print(cfg.TERMINAL_INFO, f"EPOCH {epoch+1}/{NUM_EPOCHS}")
    print(f"TRAIN loss: {avg_train_loss:.6f}, accuracy {correct_train:.4f}")
    print(f"VALIDATION loss: {avg_val_loss:.6f}, accuracy {correct_val:.4f}")

print(cfg.TERMINAL_INFO, "saving model")
torch.save(detector_model, os.path.join(cfg.SAVE_MODEL_PATH, "detector_model.path"))
