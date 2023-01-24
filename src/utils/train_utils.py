import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from PIL import Image

import utils.config as cfg


class TrainUtils:
    """
    class with utils in training process.
    """

    def __init__(self, run_name: str) -> None:
        self.run_name = run_name
        self.path = self.__run_results_path(run_name)
        self.plots_path, self.model_dics_path, self.model_path = self.__make_child_dirs(self.path)

    def __run_results_path(self, run_name: str) -> str:
        dir_path = os.path.join(cfg.RESULTS_PATH, run_name)
        self.__make_dir(dir_path)
        return dir_path

    def __make_child_dirs(self, parent_dir: str) -> tuple[str, str, str]:
        plots_path = os.path.join(parent_dir, "plots")
        model_dics_path = os.path.join(parent_dir, "model_dics")
        model_path = os.path.join(parent_dir, "model")

        paths_tuple = plots_path, model_dics_path, model_path
        for i in paths_tuple:
            self.__make_dir(i)
        return paths_tuple

    def __make_dir(self, path: str) -> None:
        if not os.path.exists(path):
            os.mkdir(path)

    def save_metric_model_dict(self):
        """
        saves dictionary wth metric model progress
        """

        pass

    def save_metric_model(self, model: nn.Module, epoch: int) -> None:
        """
        saves dictionary wth metric model progress
        """
        try:
            print(cfg.TERMINAL_INFO, "Saving model")
            torch.save(model, os.path.join(self.model_path, f"metric-{self.run_name}-e{epoch}.path"))
            print(cfg.TERMINAL_INFO, "Model saved")
        except ValueError as value_err:
            print(f"{cfg.RED}[INFO] SAFILED{cfg.RESET_COLOR}")
            print("ValueError\n", value_err)
        except Exception as ex:
            print(f"{cfg.RED}[INFO] SAVING FAILED with msg:{cfg.RESET_COLOR}")
            print(ex)

    def plot_n_save(self, val_dataset: data.Dataset, test_dataset: data.Dataset, epoch: int, dist) -> None:
        """
        plot predictions based on distances
        """
        image_size = 128
        file_path = os.path.join(self.plots_path, f"predictions-{self.run_name}-e{epoch}.png")
        plt.figure(figsize=(20, 11))
        for i, idx in enumerate(np.random.choice(len(test_dataset), size=25, replace=False)):
            matched_idx = dist[idx].argmin().item()

            actual_label = test_dataset.labels[idx]
            predicted_label = val_dataset.labels[matched_idx]

            actual_image_path = test_dataset.images[idx]
            predicted_image_path = val_dataset.images[matched_idx]

            actual_image = np.array(Image.fromarray(actual_image_path).resize((image_size, image_size)))
            predicted_image = np.array(Image.fromarray(predicted_image_path).resize((image_size, image_size)))
            stack = np.hstack([actual_image, predicted_image])

            plt.subplot(5, 5, i + 1)
            plt.imshow(stack)
            plt.title(f"Get: {actual_label}\nPreditions: {predicted_label}", fontdict={"fontsize": 8})
            plt.axis("off")
            plt.savefig(file_path)
