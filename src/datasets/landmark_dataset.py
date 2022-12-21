import os
import time
import torch
import torch.utils.data as data

# from base_dataset import BaseDataset
from PIL import Image

# import utils.config as cfg
from utils.dataset_utils import LandmarkUtils


class LandmarkDataset(data.Dataset):
    """
    Dataset class for DeepFashion Landmark Detection Dataset #4
    """

    # needs to be change to inheritance from BaseDataset
    def __init__(
        self,
        root: str,
        runing_mode: str,
        device: str,
        transforms_list: list = None,
        other_dir_name: str = None,
        load_as_tensors: bool = False,
        measure_time: bool = False,
    ) -> None:

        self.transforms = transforms_list
        self.root = root
        self.runing_mode = runing_mode
        self.device = device
        self.load_as_tensors = load_as_tensors
        self.measure_time = measure_time

        self.utils = LandmarkUtils(self.root, other_dir_name)

        self.images = list(self.utils.get_file_list(self.runing_mode))
        self.scale = self.utils.get_scale_from_dir_name(other_dir_name)

    def __getitem__(self, idx) -> tuple:
        start_time = time.time()
        index = idx % len(self.images)
        image_file_path = os.path.join(self.root, self.images[index])

        if self.load_as_tensors:
            loaded_images = torch.load(image_file_path)
        else:
            loaded_images = self._image_loader(image_path=image_file_path, image_scale=1)

        bbox = self.utils.get_bbox_position(self.images[index])
        label = (
            self.utils.get_label_type(self.images[index]) - 1
        )  # Labels are numbered in range 1-3, loss func except range 0 - (N-1)

        bbox = torch.tensor(bbox, device=self.device)
        label = torch.tensor(label, device=self.device)

        img = self.transforms(loaded_images)

        end_time = time.time()
        if self.measure_time:
            print(f"__getitem__ time  {(end_time-start_time):.2f}")
        return img, label, bbox

    def __len__(self):
        # return len of img list
        return self.utils.get_file_list(self.runing_mode).size

    def _image_loader(self, image_path: str, image_scale: float = 1) -> Image.Image:
        """
        loading image by pillow and convert it to RGB
        """
        # https://github.com/python-pillow/Pillow/issues/835

        image = Image.open(image_path)
        width, height = image.size
        newsize = (int(width * image_scale), int(height * image_scale))

        return image.resize(newsize).convert("RGB")
