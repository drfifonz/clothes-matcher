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

    def __init__(self, dataset_path: str, scale: float) -> None:
        self.dataset_path = dataset_path
        self.utils = LandmarkUtils(self.dataset_path)
        self.img_to_tensor = transforms.ToTensor()
        self.scale = scale

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

    def refactor_landmarks_file(self, landmarks_file_path, as_tensors: bool = False) -> None:
        """
        Refactor list_landmarks.txt file by changing image's file name
        """
        # TODO implement refactor_landmarks_method
        pass

    def refactor_bbox_file(self, bbox_file_path: str, as_tensors: bool = False) -> None:
        """
        Refactor list_bbox.txt file by changing image's file name and calculating bbox by scale
        """

        file_root_path, full_file_name = os.path.split(bbox_file_path)
        file_name, ext = os.path.splitext(full_file_name)

        file_name += f"_{self.scale}"
        if as_tensors:
            file_name += "_tensors"
        print(file_name)
        list_lines = self.__open_txt_file(bbox_file_path).split("\n")
        for index, line in enumerate(list_lines[2:]):

            line = line.replace("img/", f"img_{self.scale}/")
            if as_tensors:

                line = line.replace(".jpg", ".pt")

            list_line = line.split(" ")

            for line_index, coordinate in enumerate(list_line[1:]):
                list_line[line_index + 1] = f"{int(int(coordinate) * self.scale):03d}"
            line = " ".join(list_line)
            list_lines[2 + index] = line

        text = "\n".join(list_lines)

        self.__save_txt_file(os.path.join(file_root_path, file_name + ext), text)

    def __open_txt_file(self, path: str) -> str:
        with open(path, "r", encoding="utf-8") as file:
            return file.read()

    def __save_txt_file(self, path: str, text: str) -> str:
        with open(path, "w", encoding="utf-8") as file:
            return file.write(text)

    def __create_dir(self, path: str) -> None:
        if not os.path.exists(path):
            os.mkdir(path)


if __name__ == "__main__":

    SCALE = 0.5

    BBOX_PATH = "data/datasets/Deepfashion_Landmark/Anno/list_bbox.txt"

    lm_dataset_path = os.path.join(cfg.LANDMARK_DATASET_PATH, cfg.LM_IMGS_DIR_PATH)
    preprocess = Preprocess(lm_dataset_path, SCALE)

    # preprocess.save_resized_imgs(SCALE)

    preprocess.refactor_bbox_file(BBOX_PATH, True)
    # preprocess.save_imgs_as_tensors("img_0.5")
    # preprocess.save_imgs_as_tensors()
