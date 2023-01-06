import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorch_metric_learning import losses, miners
from torchvision.models import resnet50
from tqdm import tqdm
import wandb

import utils.config as cfg
from models import MetricModel
from datasets import LandmarkHDF5Dataset
from utils import ModelUtils, TrainTransforms

INITIAL_LR = 1e-3
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

# TODO UNCOMMENT WANDB init
# wandb.config = {
#     "learning_rate": INITIAL_LR,
#     "epochs": NUM_EPOCHS,
#     "batch_size": BATCH_SIZE,
#     "workers": NUM_WORKERS,
#     "gradation": GRADATION,
#     "pin_memory": PIN_MEMORY,
#     "mean": MEAN,
#     "std": STD,
# }

# wandb.init(project="clothes-matcher-metric", entity="drfifonz", config=wandb.config)

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

print(cfg.TERMINAL_INFO, f"Train photos:{len(train_dataset)}\tVal photos: {len(val_dataset)}")
resnet_utils = ModelUtils()

resnet_model = resnet_utils.build_model(model=resnet50)

model = MetricModel(resnet_model)

miner = miners.MultiSimilarityMiner()
loss_func = losses.TripletMarginLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=INITIAL_LR)

print(f"{cfg.TERMINAL_INFO} start training loop...")

for epoch in tqdm(range(NUM_EPOCHS)):
    total_train_loss = 0
    total_val_loss = 0

    for batch_idx, (image, label, _) in enumerate(train_dataloader):
        optimizer.zero_grad()

        image = image.to(DEVICE)
        label = label.to(DEVICE)

        embeddings = model(image)
        hard_pairs = miner(embeddings, label)

        loss = loss_func(embeddings, label)
        loss.backward()
        optimizer.step()
