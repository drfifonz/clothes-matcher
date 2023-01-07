import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50, resnet34, resnet18
from tqdm import tqdm
import wandb

import utils.config as cfg
from datasets.landmark_dataset import LandmarkDataset
from datasets.lm_hdf5_dataset import LandmarkHDF5Dataset
from models.detection_model import BboxDetectionModel
from utils.models_utils import ModelUtils, TrainTransforms
from utils.argument_parser import arguments_parser


args = arguments_parser()

# MEAN, STD and RESIZE_SIZE declaration
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
RESIZE_SIZE = (200, 200)
BBOX_WEIGHT = 1.0
LABEL_WEIGHT = 1.0

INITIAL_LR = args.lr
NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size

NUM_WORKERS = 8
# NUM_WORKERS = os.cpu_count()

GRADATION = True

AS_TENSOR = False
DEBUG_MODE = False


LOG_INTERVAL = 100 * 32 / BATCH_SIZE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False  # to speed up loading data on CPU to training it on GPU
# see https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723

# TODO move resnet_core to modelutils
if args.model == "resnet18":
    resnet_core = resnet18
elif args.model == "resnet34":
    resnet_core = resnet34
elif args.model == "resnet50":
    resnet_core = resnet50


wandb.config = {
    "learning_rate": INITIAL_LR,
    "epochs": NUM_EPOCHS,
    "batch_size": BATCH_SIZE,
    "workers": NUM_WORKERS,
    "gradation": GRADATION,
    "pin_memory": PIN_MEMORY,
    "mean": MEAN,
    "std": STD,
    "model": resnet_core.__name__,
}
print(wandb.config)

# wandb.init(project="clothes-matcher", entity="drfifonz", config=wandb.config)
wandb.init(project="clothes-matcher-detection", entity="drfifonz", config=wandb.config)


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


resnet_model = resnet_utils.build_model(model=resnet_core, pretrained=True, gradation=GRADATION)

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
    wandb.watch(detector_model)

    # initialization losses
    total_train_loss = 0
    total_val_loss = 0

    # initialization predictions num
    correct_train = 0
    correct_val = 0

    loading_time_start = time.time()
    for batch_idx, (image, label, bbox) in enumerate(train_dataloader):

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

        total_train_loss += total_loss
        correct_train += (predictions[1].argmax(1) == label).type(torch.float).sum().item()

        total_loss.backward()
        optimizer.step()

        if batch_idx % LOG_INTERVAL == 0:
            wandb.log(
                {
                    "iterational_train_loss": total_train_loss,
                    "iterational_train_accuracy": correct_train / len(train_dataset),
                }
            )
        it_time_end = time.time()
        if batch_idx % 10 == 0 and DEBUG_MODE:
            measured_time = it_time_end - it_time_start
            loading_time = it_time_end - loading_time_start
            print(
                f"it: {batch_idx}\t time in it: {measured_time:.2f}\t loading time per it: {(loading_time/batch_idx):.2f}",
                end="\r",
            )
    if DEBUG_MODE:
        print(end="\x1b[2K")

    with torch.no_grad():
        detector_model.eval()

        for batch_idx, (image, label, bbox) in enumerate(val_dataloader):

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

            if batch_idx % LOG_INTERVAL == 0:
                wandb.log(
                    {
                        "iterational_val_loss": total_val_loss,
                        "iterational_val_accuracy": correct_val / len(val_dataset),
                    }
                )

    avg_train_loss = total_train_loss / train_steps
    avg_val_loss = total_val_loss / val_steps

    correct_train = correct_train / len(train_dataset)
    correct_val = correct_val / len(val_dataset)

    # update training history
    progress["total_train_loss"].append(avg_train_loss.cpu().detach().numpy())
    progress["total_val_loss"].append(avg_val_loss.cpu().detach().numpy())
    progress["train_class_acc"].append(correct_train)
    progress["val_class_acc"].append(correct_val)
    wandb.log(
        {
            "total_train_loss": progress["total_train_loss"][-1],
            "total_val_loss": progress["total_val_loss"][-1],
            "train_class_acc": progress["train_class_acc"][-1],
            "val_class_acc": progress["val_class_acc"][-1],
            "epoch": epoch,
        }
    )
    #!MODEL TRAINGING INFO

    print("\n", cfg.TERMINAL_INFO, f"EPOCH {epoch+1}/{NUM_EPOCHS}")
    print(f"TRAIN loss: {avg_train_loss:.6f}, accuracy {correct_train:.4f}")
    print(f"VALIDATION loss: {avg_val_loss:.6f}, accuracy {correct_val:.4f}")

print(cfg.TERMINAL_INFO, "saving model")
torch.save(detector_model, os.path.join(cfg.SAVE_MODEL_PATH, f"detector_model_{wandb.run.name}.path"))
