import os
import sys
from tqdm import tqdm

sys.path.append("./src")

# pylint: disable=wrong-import-position

from utils.dataset_utils import LandmarkUtils
import utils.config as cfg

# pylint: enable=wrong-import-position


class Preprocess:
    """
    class for preprocessing preperation images for faster loading
    """

    def __init__(self, dataset_path) -> None:
        self.dataset_path = dataset_path
        self.utils = LandmarkUtils(self.dataset_path)

    def save_resized_imgs(self, new_path: str, image_scale: float):
        """
        saves resized images in new path
        """
        if not os.path.exists(new_path):
            os.mkdir(new_path)

        images_list = os.listdir(self.dataset_path)

        for image in tqdm(images_list):
            image_path = os.path.join(self.dataset_path, image)
            # print(image_path)
            image_pil = self.utils.image_loader(image_path, image_scale)
            image_pil.save(os.path.join(new_path, image))


if __name__ == "__main__":
    SCALE = 0.5
    path = os.path.join(cfg.LANDMARK_DATASET_PATH, cfg.LM_IMGS_DIR_PATH)
    preprocess = Preprocess(path)
    preprocess.save_resized_imgs(path + f"_{SCALE}", SCALE)
