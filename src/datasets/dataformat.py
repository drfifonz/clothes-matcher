from pathlib import Path
import sys
import os

import h5py
import lmdb
from tqdm import tqdm
import numpy as np
from PIL import Image

sys.path.append("./src")

# pylint: disable=wrong-import-position

import utils.config as cfg
from utils.dataset_utils import LandmarkUtils

# pylint: enable=wrong-import-position


class Dataformatter:
    """
    Class for converting data to minimalize i/o time in dataloader at traingn process.
    """

    def __init__(self, path: str, lmdb_path: str, hdf5_path: str) -> None:
        """
        Parameters:
        ---------------
        path: path to dataset directory
        lmdb_path: path to lmbd files
        hdf5_path: path to hdf5 files
        """

        self.path = Path(path)
        self.lmdb_path = Path(lmdb_path)
        self.hdf5_path = Path(hdf5_path)

        self.lmdb_path.mkdir(parents=True, exist_ok=True)
        self.hdf5_path.mkdir(parents=True, exist_ok=True)

        self.utils = LandmarkUtils(self.path)

    def store_images_lmdb(self, images, labels, bboxes):
        """
        Stores an array of images to lmdb.
        Parameters:
        ---------------
        images:      images array, (N, 200, 200, 3) to be stored \n
        labels:      labels array, (N, 1) to be stored \n
        bbox:        bboxes array, (N, 4) to be stored \n
        """

        pass

    def store_images_hdf5(
        self,
        mode: str,
    ):
        """
        Stores an array of images to HDF5.
        Parameters:
        ---------------
        images:      images array, (N, 200, 200, 3) to be stored \n
        labels:      labels array, (N, 1) to be stored \n
        bbox:        bboxes array, (N, 4) to be stored \n
        """
        images, labels, bboxes = self.__get_images_data(mode)
        # num_images = len(images)

        # create hdf5 file
        file = h5py.File(self.hdf5_path / f"images_{mode}.h5", "w")

        # Create a dataset in the file
        file.create_dataset("images", np.shape(images), h5py.h5t.STD_U8BE, data=images)
        file.create_dataset("labels", np.shape(labels), h5py.h5t.STD_U8BE, data=labels)
        file.create_dataset("bboxes", np.shape(bboxes), h5py.h5t.STD_U8BE, data=bboxes)

        file.close()

    def __get_images_data(self, mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        returns all images with their labels and bboxes positions as tuple of np.ndarrays
        """
        size = 200, 200

        image_list = list(self.utils.get_file_list(mode))
        list_photos = []
        list_labels = []
        list_bboxes = []

        print("Stacking images data as np.array")
        for image in tqdm(image_list):

            img_path = os.path.join(self.path, image)
            image_array, im_size = self.__get_single_image(img_path, size)

            image_array = np.expand_dims(image_array, axis=0)

            scale = size[0] / im_size[0]
            # it is possible to subtract 1 at label due to numerating from 0 not from 1

            label_array = self.utils.get_label_type(image).reshape(1, 1)
            bbox_scaled = list(map(lambda cord: int(cord * scale), self.utils.get_bbox_position(image)))
            bbox_array = np.array(bbox_scaled).reshape(1, 4)

            list_photos.append(image_array)
            list_labels.append(label_array)
            list_bboxes.append(bbox_array)

        images = np.vstack(list_photos)
        labels = np.vstack(list_labels)
        bboxes = np.vstack(list_bboxes)
        print(images.shape)
        print(labels.shape)
        print(bboxes.shape)
        return images, labels, bboxes

    def __get_single_image(self, image_path: str, size: tuple[int, int]) -> np.ndarray:
        """
        Returns image as numpy array.
        """
        image = Image.open(image_path)
        im_size = image.size
        return np.asarray(image.resize(size)), im_size


if __name__ == "__main__":
    IMG_DIR_PATH = "data/datasets/Deepfashion_Landmark"
    LMDB_DIR_PATH = "data/datasets/Deepfashion_Landmark/img_lmdb"
    HDF5_DIR_PATH = "data/datasets/Deepfashion_Landmark/img_hdf5"
    SINGLE_IMG_PATH1 = "data/datasets/Deepfashion_Landmark/img/img_00000001.jpg"
    SINGLE_IMG_PATH2 = "data/datasets/Deepfashion_Landmark/img/img_00000002.jpg"

    df = Dataformatter(path=IMG_DIR_PATH, lmdb_path=LMDB_DIR_PATH, hdf5_path=HDF5_DIR_PATH)

    # df.__get_images_data("train")
    # df.store_images_hdf5("train")
    print("DATASET TYPE:\t\t val")
    df.store_images_hdf5("val")
    print("DATASET TYPE:\t\t test")
    df.store_images_hdf5("test")
