import os
import torch
from torchvision.transforms import transforms
from PIL import Image
import utils.config as cfg


class ModelUtils:
    """
    Utils class for models
    """

    def __init__(self, model) -> None:
        self.model = model

    def save_model(self, epochs, optimizer, criterion):
        """
        save trained model
        """
        torch.save(
            {
                "epochs": epochs,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": criterion,
            },
            os.path.join(cfg.SAVE_MODEL_PATH, f"model_e{epochs+1}.pth"),
        )

    def build_model(self, pretrained: bool = True, gradation: bool = True):
        """
        Building model with pretreined parameter and gradation info.
        """
        if pretrained:
            print(f"{cfg.TERMINAL_INFO}loading model with pretrained weigts.")
        else:
            print(f"{cfg.TERMINAL_INFO} loading model {cfg.RED}without{cfg.RESET_COLOR} pretrained weigts.")
        if gradation:
            print(f"{cfg.TERMINAL_INFO} Require gradation for all layers.")
        else:
            print(f"{cfg.TERMINAL_INFO} Freezing all hiden layers.")

        self.model = self.model(pretrained=pretrained)
        for parameters in self.model.parameters():
            parameters.requires_grad = gradation
        return self.model


class TrainTransforms:
    """
    Transforms class for train dataset
    """

    def __init__(self, mean: tuple, std: tuple, resize_size: tuple[int] = None) -> None:
        transformation_list = []
        if resize_size:
            transformation_list.append(transforms.Resize(resize_size))

        transformation_list += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]

        self.transforms = transforms.Compose(transformation_list)
        print(self.transforms)

    def __call__(self, img: Image):
        return self.transforms(img=img)


class ValidTransforms:
    """
    Transforms class for validation dataset
    """

    def __init__(self, mean: tuple, std: tuple, resize_size: tuple[int] = None) -> None:
        transformation_list = []
        if resize_size:
            transformation_list.append(transforms.Resize(resize_size))

        transformation_list += [
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]

        self.transforms = transforms.Compose(transformation_list)
        print(self.transforms)

    def __call__(self, img: Image):
        return self.transforms(img=img)
