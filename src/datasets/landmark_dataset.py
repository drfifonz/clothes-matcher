import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
from base_dataset import BaseDataset


class LandmarkDataset(data.Dataset):
    """
    Dataset class for DeepFashion Landmark Detection Dataset #4
    """

    # needs to be change to inheritance from BaseDataset
    def __init__(self, root: str, transforms_list: list = None) -> None:
        if not transforms_list:
            raise ValueError("No tranforms_list")
        self.root = root
        self.transform = transforms.Compose(transforms_list)

    def __getitem__(self, index) -> dict:
        return super().__getitem__(index)

    def __len__(self):
        pass
        # return len of img list

    def _image_loader(self, image_path: str, image_scale: float = 1) -> Image.Image:
        """
        loading image by pillow and convert it to RGB
        """
        # https://github.com/python-pillow/Pillow/issues/835

        image = Image.open(image_path)
        width, height = image.size
        newsize = (int(width * image_scale), int(height * image_scale))

        return image.resize(newsize).convert("RGB")
