# import os
import time
from pathlib import Path
import numpy as np

import torch
import torch.utils.data as data

import utils.config as cfg
from utils.dataset_utils import LandmarkUtils


class LandmarkHDF5Dataset(data.Dataset):
    """
    Dataset class for DeepFashion Landmark Detection Dataset #4
    stored as HDF5 .h5 files splitted for train,val,test files

    Params:
    ------
    root: <str> \t path to hdf5 direrctory \n
    running_mode: <str> \t train | val | test \n
    transforms_list: <list> \t list of transform.Compose operation applied on image \n
    TODO: measure_time <bool> : decides to print loading image time
    """

    def __init__(
        self,
        root: str,
        running_mode: str,
        transforms_list: list = None,
        measure_time: bool = False,
        is_metric: bool = False,
    ) -> None:
        self.root = Path(root)
        self.running_mode = running_mode
        self.transforms = transforms_list
        self.measure_time = measure_time
        self.is_metric = is_metric
        self.utils = LandmarkUtils(cfg.LANDMARK_DATASET_PATH)

        self.images, self.labels, self.bboxes = self.__load_data()

    def __getitem__(self, idx) -> tuple:
        index = idx % self.images.shape[0]
        # TODO add transforms to image
        # image = self.transforms(self.images[index])
        img = self.images[index]

        img = img.reshape(3, 200, 200).astype("float32")
        # img = np.array([img[2], img[0], img[1]]).astype("float32")

        image = torch.from_numpy(img)
        label = torch.from_numpy(self.labels[index] - 1)
        #! Labels are numbered in range 1-3, loss func except range 0 - (N-1)

        bbox = torch.from_numpy(self.bboxes[index]).type(torch.int64)
        # temp = "JEST GIT " if not self.is_metric else "cos nie tak"

        # print(self.is_metric)
        res = (image, label[0], bbox) if not self.is_metric else (image, label[0])
        # res = "(image, label[0], bbox)" if not self.is_metric else "(image, label)"
        # print(len(res))
        return res
        # return image, label[0], bbox

    def __load_data(self) -> tuple[np.ndarray, np.ndarray, np.array]:
        hdf5_file_path = self.__get_file_path()
        start_time = time.time()
        dataset = self.utils.read_hdf5_dataset_file(hdf5_file_path)
        stop_time = time.time()
        print(
            f"{cfg.TERMINAL_INFO} {Path(hdf5_file_path).name} file loaded",
            f"in {(stop_time-start_time):.2f}s." if self.measure_time else "",
        )
        return dataset

    def __get_file_path(self) -> str:

        file_path = list(self.root.glob(f"*_{self.running_mode}.h5"))
        return file_path[0]

    def __len__(self):
        # return len of images
        return self.images.shape[0]
