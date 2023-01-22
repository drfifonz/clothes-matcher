import logging
import os

import numpy as np

# from torch.utils.data import DataLoader
import pytorch_metric_learning
import torch
import torch.nn as nn
import umap
from pytorch_metric_learning import losses, miners, samplers, testers, trainers
from pytorch_metric_learning.utils import common_functions, logging_presets
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights
from tqdm import tqdm

import utils.config as cfg
import wandb
from datasets import LandmarkHDF5Dataset
from models import Embedder
from utils import ModelUtils, TrainTransforms

print("IMPORTS DONE")

logging.getLogger().setLevel(logging.INFO)


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


# train_dataloader = DataLoader(
#     dataset=train_dataset,
#     shuffle=True,
#     batch_size=BATCH_SIZE,
#     num_workers=NUM_WORKERS,
#     pin_memory=PIN_MEMORY
# )

# val_dataloader = DataLoader(
#     dataset=val_dataset,
#     shuffle=True,
#     batch_size=BATCH_SIZE,
#     num_workers=NUM_WORKERS,
#     pin_memory=PIN_MEMORY
# )
# pretrained= True is deprecated!
trunk = resnet50(weights=ResNet50_Weights.DEFAULT)
trunk_output_size = trunk.fc.in_features
trunk.fc = common_functions.Identity()
trunk = torch.nn.DataParallel(trunk.to(DEVICE))

embedder = torch.nn.DataParallel(Embedder([trunk_output_size, 64]).to(DEVICE))

# optimizers
trunk_optimizer = torch.optim.Adam(trunk.parameters(), lr=INITIAL_LR, weight_decay=0.0001)
embedder_optimizer = torch.optim.Adam(embedder.parameters(), lr=INITIAL_LR, weight_decay=0.0001)

# loss function
loss_fun = losses.TripletMarginLoss(margin=0.1)

# mining function
miner_fun = miners.MultiSimilarityMiner(epsilon=0.1)

# dataloader sampler
sampler = samplers.MPerClassSampler(
    labels=np.squeeze(train_dataset.labels), m=4, length_before_new_iter=len(train_dataset)
)


models = {"trunk": trunk, "embedder": embedder}
optimizers = {"trunk_optimizer": trunk_optimizer, "embedder_optimizer": embedder_optimizer}
loss_fun = {"metric_loss": loss_fun}
mining_fun = {"tuple_miner": miner_fun}


# keeps records from training in given folders as csv files
# pip install record-keeper, pip install tensorboard

record_keeper, _, _ = logging_presets.get_record_keeper(
    "data/training_logs", "data/training_tensorboards"
)

hooks = logging_presets.get_hook_container(record_keeper)
dataset_dict = {"val": val_dataset}
saved_models_path = "data/saved_models"

# testers need faiss:
# conda install -c pytorch faiss-cpu 
# or conda install -c pytorch faiss-gpu (contains CPU and GPU support)

# if the above fail, use conda-forge: 
# conda install -c conda-forge faiss-cpu or
# conda install -c conda-forge faiss-gpu
tester = testers.GlobalEmbeddingSpaceTester(
    end_of_testing_hook=hooks.end_of_testing_hook,
    visualizer=umap.UMAP(),
)
print("IM HERE")
# TODO add hooks, a tester and a trainer
