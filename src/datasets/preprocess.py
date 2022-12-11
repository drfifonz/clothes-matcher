import os
import sys

import torch
from torchvision import transforms
from tqdm import tqdm

sys.path.append("./src")

# pylint: disable=wrong-import-position

import utils.config as cfg
from utils.dataset_utils import LandmarkUtils

# pylint: enable=wrong-import-position


class Preprocess:
    """
    class for preprocessing preperation images for faster loading
    """

    def __init__(self, dataset_path) -> None:
        self.dataset_path = dataset_path
        self.utils = LandmarkUtils(self.dataset_path)
        self.img_to_tensor = transforms.ToTensor()

    def save_resized_imgs(self, image_scale: float) -> None:
        """
        saves all resized images from dataset folder in new dir
        """
        new_path = self.dataset_path + f"_{image_scale}"

        self.__create_dir(new_path)

        images_list = os.listdir(self.dataset_path)

        for image in tqdm(images_list):
            image_path = os.path.join(self.dataset_path, image)
            image_pil = self.utils.image_loader(image_path, image_scale)

            image_pil.save(os.path.join(new_path, image))

    def save_imgs_as_tensors(self, dir_name: str = None) -> None:
        """
        save all images form dataset folder  as tensors in .pt format\n
        dir_name is OPTIONAL parameter to loading imgs from another directory than default dataset folder
        """

        head_path, tail_path = os.path.split(self.dataset_path)
        tail_path = dir_name if dir_name else tail_path
        dataset_path = os.path.join(head_path, tail_path)
        new_path = dataset_path + "_tensors"

        self.__create_dir(new_path)

        images_list = os.listdir(dataset_path)

        for image in tqdm(images_list):
            image_path = os.path.join(dataset_path, image)
            img_name, _ = os.path.splitext(image)
            image_pil = self.utils.image_loader(image_path)

            img_tensor = self.img_to_tensor(image_pil)
            img_tensor_path = os.path.join(new_path, img_name + ".pt")

            torch.save(img_tensor, img_tensor_path)

    def __create_dir(self, path: str) -> None:
        if not os.path.exists(path):
            os.mkdir(path)


if __name__ == "__main__":

    SCALE = 0.5

    lm_dataset_path = os.path.join(cfg.LANDMARK_DATASET_PATH, cfg.LM_IMGS_DIR_PATH)
    preprocess = Preprocess(lm_dataset_path)

    # preprocess.save_resized_imgs(SCALE)

    preprocess.save_imgs_as_tensors("img_0.5")
    # preprocess.save_imgs_as_tensors()
