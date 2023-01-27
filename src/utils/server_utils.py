import base64
import io

# import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

import utils.config as cfg


class ServerUtils:
    """
    Helper class for server
    """

    def __init__(self) -> None:
        pass

    def image_pil_to_base64(self, image: Image.Image) -> str:
        """
        Converts a PIL Image to base64 string
        """
        buffer = io.BytesIO()
        # image.save(buffer, format=image.format)
        buffer = buffer.getvalue()
        return base64.b64encode(buffer).decode("utf-8")

    def image_base64_to_pil(self, img_string: str) -> Image.Image:
        """
        convert string in base64 format to PIL image
        """
        img_bytes = base64.b64decode(img_string)
        img_pil = Image.open(io.BytesIO(img_bytes))
        return img_pil

    def combine_images(self, images: list[Image.Image], show_image: bool = False) -> Image.Image:
        """
        (for local tests only) stack images to one horizontaly
        """
        resized_images_list = list(map(lambda photo: photo.resize((300, 500)), images))
        np_images_list = [np.array(image) for image in resized_images_list]
        stacked_image = np.hstack(np_images_list)
        image = Image.fromarray(stacked_image)
        if show_image:
            image.show()

        return image

    def __get_random_images_paths(self, num_of_images: int) -> list[str]:
        """
        return paths to random images from landmark dataset img dir
        """
        dir_path = os.path.join(cfg.LANDMARK_DATASET_PATH, cfg.LM_IMGS_DIR_PATH)
        images = random.choices(os.listdir(dir_path), k=num_of_images)
        images_paths = [os.path.join(dir_path, im) for im in images]

        return images_paths

    def get_random_pil_images(self, num_of_images: int) -> list[Image.Image]:

        """
        return list of random PIL images  from dataset
        """
        paths = self.__get_random_images_paths(num_of_images)

        return [Image.open(path) for path in paths]

    def image_pil_to_tensor(self, pil_image: Image.Image) -> torch.Tensor:
        """
        converts PIL image to tensor acceptable by model
        """
        image_array = np.asarray(pil_image)
        # print("im arr:", image_array.shape)
        image_array = np.expand_dims(image_array, axis=0)
        # print("im arr after expands:", image_array.shape)
        # print("image_array[0] shape:", image_array[0].shape)

        img = image_array[0].reshape(3, 200, 200).astype("float32")
        # print("img.shape: ", img.shape)
        image_tensor = torch.from_numpy(img)
        return image_tensor

    def image_tensor_to_numpy(self, tensor_image) -> np.ndarray:
        """
        converts tensor image to numpy array
        """
        # print(tensor_image.shape)
        # print(tensor_image.shape)
        tensor_image = tensor_image.reshape(tensor_image.shape[1], tensor_image.shape[2], tensor_image.shape[0])
        return (tensor_image.numpy()).astype(np.uint8)

    def image_numpy_to_pil(self, numpy_image: np.ndarray) -> Image.Image:
        """
        converts numpy ndarary image to PIL image
        """
        return Image.fromarray(numpy_image)

    def imshow(self, images_list: list[np.ndarray], figsize=(8, 4)) -> None:
        """
        show stacked images in np.darray format
        for debug purpouse
        """
        images = np.hstack(images_list)
        plt.figure(figsize=figsize)
        plt.imshow(images)
        plt.show()

    def load_model(self, model_path: str, device: str) -> nn.Module:
        """
        loads pretrained model and set its mode to eval
        """
        device = torch.device(device=device)
        model = torch.load(model_path)
        model.to(device)
        model.eval()
        return model
