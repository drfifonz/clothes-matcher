import os

# import matplotlib.pyplot as plt
import numpy as np
import torch

import wandb

# from PIL import Image
from pytorch_metric_learning import distances, losses, miners

# import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from tqdm import tqdm

import utils.config as cfg
from datasets import LandmarkHDF5Dataset
from models import MetricModel
from utils import ModelUtils, TrainTransforms, TrainUtils

INITIAL_LR = 1e-3
NUM_EPOCHS = 20
BATCH_SIZE = 24

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

IMAGE_SIZE = 128
SAVE_FREQ = 2

EMBEDDING_SIZE = 100
GRADATION = True
LOG_INTERVAL = 100
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
    is_metric=True,
)
val_dataset = LandmarkHDF5Dataset(
    root=cfg.HDF5_DIR_PATH,
    running_mode="val",
    transforms_list=tf_list,
    measure_time=False,
    is_metric=True,
)
# test_dataset = LandmarkHDF5Dataset(
#     root=cfg.HDF5_DIR_PATH,
#     running_mode="test",
#     transforms_list=tf_list,
#     measure_time=False,
#     is_metric=True,
# )

train_dataloader = DataLoader(
    dataset=train_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
)

# val_dataloader = DataLoader(
#     dataset=val_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
# )

print(cfg.TERMINAL_INFO, f"Train photos:{len(train_dataset)}\tVal photos: {len(val_dataset)}")

model_utils = ModelUtils()

resnet_model = model_utils.build_model(model=resnet50)

model = (
    MetricModel(resnet_model, embedding_size=EMBEDDING_SIZE).cuda()
    if DEVICE == "cuda"
    else MetricModel(resnet_model, embedding_size=EMBEDDING_SIZE)
)

distance = distances.CosineSimilarity()

miner = miners.TripletMarginMiner(margin=0.2, distance=distance, type_of_triplets="semihard")
# miner = miners.MultiSimilarityMiner()
loss_func = losses.TripletMarginLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=INITIAL_LR)

print(f"{cfg.TERMINAL_INFO} initializing {cfg.RED}wandb{cfg.RESET_COLOR}")

wandb.config = {
    "learning_rate": INITIAL_LR,
    "epochs": NUM_EPOCHS,
    "batch_size": BATCH_SIZE,
    "workers": NUM_WORKERS,
    "gradation": GRADATION,
    "pin_memory": PIN_MEMORY,
    "mean": MEAN,
    "std": STD,
    "embedding_size": EMBEDDING_SIZE,
    "miner": miner._get_name(),
    "distance": distance._get_name(),
}

wandb.init(project="clothes-matcher-metric-temp", entity="drfifonz", config=wandb.config)

train_utils = TrainUtils(run_name=wandb.run.name)


print(f"{cfg.TERMINAL_INFO} start training loop...")

for epoch in tqdm(range(NUM_EPOCHS)):
    # total_train_loss = 0
    # total_val_loss = 0

    model.train()
    # WANDB
    wandb.watch(model)

    # train loop
    # for batch_idx, (image, label, _) in enumerate(train_dataloader):
    for batch_idx, (image, label) in enumerate(train_dataloader):
        image = image.to(DEVICE)
        label = label.to(DEVICE)

        optimizer.zero_grad()

        embeddings = model(image)
        hard_pairs = miner(embeddings, label)

        loss = loss_func(embeddings, label, hard_pairs)
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print(
                f"Epoch: {epoch}, Iterations: {batch_idx},\t",
                f"Loss: {loss:.6f},\t Mined triplets: {miner.num_triplets}",
                end="\r",
            )
            wandb.log(
                {
                    "loss": loss,
                    "mined_triplets": miner.num_triplets,
                }
            )
    # print(f"epoch loop done {epoch}")
    print(end="\x1b[2K")
    print("VALIDATION")
    model.eval()

    with torch.no_grad():

        train_embeddings, _ = model_utils.get_all_embeddings(train_dataset, model, device=DEVICE)
        val_embeddings, _ = model_utils.get_all_embeddings(val_dataset, model, device=DEVICE)

        # for each val embedding, find distance with all embeddings in train embeddings
        dist = torch.cdist(val_embeddings.cpu(), train_embeddings.cpu())
        query_labels = np.array(val_dataset.labels)

        # Find index of closesest matching embedding
        matched_idx = torch.argmin(dist, axis=1).cpu().numpy()
        matched_labels = np.array(train_dataset.labels)[matched_idx]

        accuracy = (query_labels == matched_labels).mean()

        # accuracy = get_accuracy(val_dataset, train_dataset, model, DEVICE)

        print(f"Accuracy: {accuracy}")
        wandb.log(
            {
                "epoch": epoch,
                "accuracy": accuracy,
            }
        )

    print("VISUALIZATION")
    # plt.figure(figsize=(20, 11))
    # for i, idx in enumerate(np.random.choice(len(val_dataset), size=25, replace=False)):
    #     matched_idx = dist[idx].argmin().item()

    #     actual_label = val_dataset.labels[idx]
    #     predicted_label = train_dataset.labels[matched_idx]

    #     actual_image = val_dataset.images[idx]
    #     predicted_image = train_dataset.images[matched_idx]

    #     actual_image = np.array(Image.fromarray(actual_image).resize((IMAGE_SIZE, IMAGE_SIZE)))
    #     predicted_image = np.array(Image.fromarray(predicted_image).resize((IMAGE_SIZE, IMAGE_SIZE)))

    #     stack = np.hstack([actual_image, predicted_image])

    #     plt.subplot(5, 5, i + 1)
    #     plt.imshow(stack)
    #     plt.title(f"Get: {actual_label}\nPrediction: {predicted_label}", fontdict={"fontsize": 8})
    #     plt.axis("off")
    #     # plt.show()

    if epoch % SAVE_FREQ == 0:
        train_utils.save_metric_model(model, epoch)
        train_utils.plot_n_save(
            val_dataset=train_dataset,
            test_dataset=val_dataset,
            epoch=epoch,
            dist=dist,
        )
    del dist, matched_idx
    del val_embeddings, train_embeddings
    torch.cuda.empty_cache()
