import os
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
    ) -> None:

        self.transforms = transforms_list
        # self.transforms = transforms.Compose(transforms_list)

        self.root = root
        self.runing_mode = runing_mode
        self.device = device
        self.utils = LandmarkUtils(self.root)

        self.images = list(self.utils.get_file_list(self.runing_mode))

    def __getitem__(self, idx) -> tuple:
        index = idx % len(self.images)
        image_path = os.path.join(self.root, self.images[index])
        # print(image_path)
        image_pil = self._image_loader(image_path=image_path, image_scale=1)

        bbox = self.utils.get_bbox_position(self.images[index])
        label = self.utils.get_label_type(self.images[index])

        bbox = torch.tensor(bbox, device=self.device)
        label = torch.tensor(label, device=self.device)

        # print("inside ok")
        if self.transforms is None:
            raise TypeError("Transforms list is not defined")
        else:
            img = self.transforms(image_pil)

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
