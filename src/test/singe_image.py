# pylint: skip-file
import sys
import numpy as np

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
from torchvision.transforms import transforms, functional
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.utils.inference import InferenceModel, MatchFinder
from PIL import Image
import time

sys.path.append("./src")
import utils.config as cfg
from datasets import LandmarkHDF5Dataset
from utils import TrainTransforms

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def convert_pil_to_tensor(pil_img):
    image_array = np.asarray(pil_img)
    print("im arr:", image_array.shape)
    image_array = np.expand_dims(image_array, axis=0)
    print("im arr after expands:", image_array.shape)
    print("image_array[0] shape:", image_array[0].shape)

    img = image_array[0].reshape(3, 200, 200).astype("float32")
    print("img.shape: ", img.shape)
    image_tensor = torch.from_numpy(img)
    return image_tensor


tf_list = TrainTransforms(mean=MEAN, std=STD, as_hdf5=True)


def image_tensor_to_numpy(tensor_image):
    print(1111111111111111111)
    # print(tensor_image.shape)
    # print(tensor_image.shape)
    tensor_image = tensor_image.reshape(tensor_image.shape[1], tensor_image.shape[2], tensor_image.shape[0])
    return (tensor_image.numpy()).astype(np.uint8)


def imshow(img, figsize=(8, 4)):
    # img = inv_normalize(img)
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.show()


print("dataset")

# train_dataset = LandmarkHDF5Dataset(
#     root=cfg.HDF5_DIR_PATH,
#     running_mode="train",
#     transforms_list=tf_list,
#     measure_time=False,
#     is_metric=True,
# )

print("dataset loaded")
# im_path = "data/temp/garniak.jpg"
im_path = "data/temp/tshirt.jpg"
image = Image.open(im_path)
image = image.resize((200, 200)).convert("RGB")
im_type = 2
img_type = torch.tensor(im_type)
#####################################

input_image_tensor = convert_pil_to_tensor(image)
input_im_array = image_tensor_to_numpy(input_image_tensor)

# temp = Image.fromarray(image_tensor_to_numpy(input_image_tensor))
# temp.show()
# raise
#####################################
model_path = "data/results/golden-laughter-3/model/metric-golden-laughter-3-e9.path"
device = torch.device("cuda")
model = torch.load(model_path)
model.to(device)
model.eval()

print("model loaded")


match_finder = MatchFinder(distance=CosineSimilarity(), threshold=0.7)
inference_model = InferenceModel(model, match_finder=match_finder)


img = torch.unsqueeze(input_image_tensor, 0)
# img = [img_type, img[0]]
print(img.shape)
inference_model.train_knn(train_dataset)
start = time.time()
distances, indices = inference_model.get_nearest_neighbors(img.float(), k=10)
print(type(distances))
print(distances)
nearest_imgs = [convert_image(train_dataset[i][0]) for i in indices.cpu()[0]]
stop = time.time()
print(f"TIME : {(stop-start):.2f}")

res_imgs = np.hstack([input_im_array, *nearest_imgs])
imshow(res_imgs)
