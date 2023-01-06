import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorch_metric_learning import losses, miners
from torchvision.models import resnet50
from tqdm import tqdm
import wandb

import utils.config as cfg
from datasets import LandmarkHDF5Dataset
from utils import ModelUtils, TrainTransforms

INITIAL_LR = 1e-4
NUM_EPOCHS = 20
BATCH_SIZE = 32

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


GRADATION = True

LOG_INTERVAL = 100 * 32 / BATCH_SIZE
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False  # to speed up loading data on CPU to training it on GPU
# see https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723
NUM_WORKERS = os.cpu_count()
# raise
wandb.config = {
    "learning_rate": INITIAL_LR,
    "epochs": NUM_EPOCHS,
    "batch_size": BATCH_SIZE,
    "workers": NUM_WORKERS,
    "gradation": GRADATION,
    "pin_memory": PIN_MEMORY,
    "mean": MEAN,
    "std": STD,
}

wandb.init(project="clothes-matcher-metric", entity="drfifonz", config=wandb.config)

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

train_dataloader = DataLoader(
    dataset=train_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
)

val_dataloader = DataLoader(
    dataset=val_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
)
