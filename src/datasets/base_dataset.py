import torch
import torchvision.transforms as transforms


class BaseDataset(torch.utils.data.Dataset):
    """
    Base dataset class used to inheritance by specific ones
    """

    def __init__(self, root: str, transforms_list: list = None) -> None:
        if not transforms_list:
            raise ValueError("No tranforms_list")
        self.transform = transforms.Compose(transforms_list)

    def __getitem__(self, index):
        pass
