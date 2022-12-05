import io
import os

import numpy as np
import pandas as pd

import utils.config as cfg


class LandmarkUtils:

    """
    utils class for Landmark Decetion dataset
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self.type_path = os.path.join(self.path, cfg.LM_TYPE_PARTITION_FILE_PATH)
        self.bbox_path = os.path.join(self.path, cfg.LM_BBOX_FILE_PATH)
        self.eval_path = os.path.join(self.path, cfg.LM_EVAL_FILE_PATH)
        self.landmarks_path = os.path.join(self.path, cfg.LM_LANDMARSKS_FILE_PATH)

    def get_file_list(self, running_mode: str) -> pd.DataFrame:
        """
        gets file list for specified runnig mode.
        Acceptable runnig modes: train / test / val
        """

        data = self.__load_evaluation_list(self.eval_path)

        match running_mode:
            case "train":
                data = data[data["evaluation_status"] == running_mode]

                return data["image_name"]

            case "test":
                data = data[data["evaluation_status"] == running_mode]

                return data["image_name"]
            case "val":
                data = data[data["evaluation_status"] == running_mode]

                return data["image_name"]
            case _:
                print("No accetable runnig mode for selecting dataset.")

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
