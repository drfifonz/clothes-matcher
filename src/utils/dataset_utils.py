import io
import os

import h5py
import numpy as np
import pandas as pd
from PIL import Image

import utils.config as cfg


class LandmarkUtils:

    """
    utils class for Landmark Decetion dataset
    """

    def __init__(self, path: str, other_dir_name: str = None) -> None:
        self.path = path

        if other_dir_name == "img_0.5":
            self.bbox_file = cfg.LM_BBOX_FILE_PATH_05
            self.eval_file = cfg.LM_EVAL_FILE_PATH_05
            self.landmarks_file = cfg.LM_LANDMARSKS_FILE_PATH_05

        elif other_dir_name == "img_0.5_tensors":
            self.bbox_file = cfg.LM_BBOX_FILE_PATH_05_TENSORS
            self.eval_file = cfg.LM_EVAL_FILE_PATH_05_TENSORS
            self.landmarks_file = cfg.LM_LANDMARSKS_FILE_PATH_05_TENSORS

        else:
            self.bbox_file = cfg.LM_BBOX_FILE_PATH
            self.eval_file = cfg.LM_EVAL_FILE_PATH
            self.landmarks_file = cfg.LM_LANDMARSKS_FILE_PATH

        # self.type_path = os.path.join(self.path, cfg.LM_TYPE_PARTITION_FILE_PATH)
        self.bbox_path = os.path.join(self.path, self.bbox_file)
        self.eval_path = os.path.join(self.path, self.eval_file)
        self.landmarks_path = os.path.join(self.path, self.landmarks_file)

    def get_file_list(self, running_mode: str) -> pd.DataFrame:
        """
        gets file list for specified runnig mode.
        Acceptable runnig modes: train / test / val
        """

        data = self.__load_evaluation_list(self.eval_path)

        match running_mode:
            case "train":
                data = data[data["evaluation_status"] == running_mode]
            case "test":
                data = data[data["evaluation_status"] == running_mode]
            case "val":
                data = data[data["evaluation_status"] == running_mode]
            case _:
                print("No accetable runnig mode for selecting dataset.")

        return data["image_name"]

    def get_bbox_position(self, image_name: str) -> list:
        """
        Gets X,Y position of BBOX returned as list of codinates as [X1,Y1,X2,Y2]
        """
        bbox_list = self.__load_bbox_list(self.bbox_path)
        image_data = bbox_list[bbox_list["image_name"] == image_name]

        bbox_position = list(image_data.iloc[0])[1 : len(image_data.iloc[0])]

        return bbox_position

    def get_landmarks(self, image_name: str) -> pd.DataFrame:
        """
        Gets landmarsk position data
        """

        lm_list = self.__load_landmarks_list(self.landmarks_path)
        image_data = lm_list[lm_list["image_name"] == image_name]

        return image_data

    def get_label_type(self, image_name: str) -> np.int64:
        """
        gets label value for image, return as numpy int 32
        """

        return self.get_landmarks(image_name).iloc[0][["clothes_type"]][
            0
        ]  # last 0 is for unpacking from pd.DataFrame to np.int64

    def __load_evaluation_list(self, path: str) -> pd.DataFrame:
        data = pd.read_csv(path, sep=" ", skiprows=[0, 1], names=["image_name", "evaluation_status"])

        return data

    def __load_bbox_list(self, path: str) -> pd.DataFrame:
        buffer_data = self.__open_txt_file(path)

        return pd.read_csv(filepath_or_buffer=buffer_data, sep=" ", skiprows=[0])

    def __load_landmarks_list(self, path: str) -> pd.DataFrame:
        buffer_data = self.__open_txt_file(path)

        return pd.read_csv(filepath_or_buffer=buffer_data, sep=" ", skiprows=[0])

    def __open_txt_file(self, path: str) -> io.StringIO:

        with open(path, "r", encoding="utf-8") as file:
            raw_data = file.read()
            raw_data = raw_data.replace("  ", " ")
            buffer_data = io.StringIO(raw_data)
        return buffer_data

    def image_loader(self, image_path: str, image_scale: float = 1) -> Image.Image:
        """
        loading image by pillow and convert it to RGB
        """
        # https://github.com/python-pillow/Pillow/issues/835

        image = Image.open(image_path)
        width, height = image.size
        newsize = (int(width * image_scale), int(height * image_scale))

        return image.resize(newsize).convert("RGB")

    def get_scale_from_dir_name(self, dir_name: str) -> float:
        """
        parse directory name to get scale of saved images
        """
        if not dir_name:
            return 1
        splitted_name = dir_name.split("_")
        try:
            float(splitted_name[1])
        except ValueError:
            print("There is no scale in directory name")

    def read_hdf5_dataset_file(self, file_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        read images, labels & bboxex and return it as tuple

        Returns:
        ---------------
        images:      images np.ndarray size: (N, 200, 200, 3) \n
        labels:      labels np.ndarray size: (N, 1)  \n
        bboxes:        bboxes np.ndarraysize: (N, 4) \n

        """
        # images, labels, bboxes = [], [], []

        # read  hdf5 file
        file = h5py.File(file_path, "r+")

        images = np.array(file["/images"])
        labels = np.array(file["/labels"])
        bboxes = np.array(file["/bboxes"])

        return images, labels, bboxes
