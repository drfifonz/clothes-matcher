import os
import torch
from torchvision.transforms import transforms
import torch.nn as nn
import utils.config as cfg
from pytorch_metric_learning import testers
import torch.utils.data.dataset as Dataset


class ModelUtils:
    """
    Utils class for models
    """

    def __init__(self) -> None:
        pass

    def save_model(self, model, epochs, optimizer, criterion):
        """
        save trained model
        """
        torch.save(
            {
                "epochs": epochs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": criterion,
            },
            os.path.join(cfg.RESULTS_PATH, f"model_e{epochs+1}.pth"),
        )

    def build_model(self, model, pretrained: bool = True, gradation: bool = True):
        """
        Building model with pretreined parameter and gradation info.
        """
        if pretrained:
            print(f"{cfg.TERMINAL_INFO}loading model with pretrained weigts.")
            model = model(weights="DEFAULT")
        else:
            print(f"{cfg.TERMINAL_INFO} loading model {cfg.RED}without{cfg.RESET_COLOR} pretrained weigts.")
        if gradation:
            print(f"{cfg.TERMINAL_INFO} Require gradation for all layers.")
        else:
            print(f"{cfg.TERMINAL_INFO} Freezing all hiden layers.")

        for parameters in model.parameters():
            parameters.requires_grad = gradation
        return model

    def get_all_embeddings(self, dataset, model: nn.Module, device):
        tester = testers.BaseTester(data_device=device)
        # return tester.get_all_embeddings(dataset, model.to(device))
        res = tester.get_all_embeddings(dataset, model.to(device), return_as_numpy=False)
        del tester
        return res
        # return tester.get_all_embeddings(dataset, model.to(device), return_as_numpy=False)


class TrainTransforms:
    """
    Transforms class for train dataset
    """

    def __init__(
        self, mean: tuple, std: tuple, resize_size: tuple[int] = None, as_tensor: bool = False, as_hdf5: bool = False
    ) -> None:
        transformation_list = []
        if as_hdf5:
            transformation_list += [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]
        else:

            if resize_size:
                transformation_list.append(transforms.Resize(resize_size))

            transformation_list += [
                # transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]
            if not as_tensor:
                transformation_list.append(transforms.ToTensor())

        transformation_list.append(transforms.Normalize(mean=mean, std=std))

        self.transforms = transforms.Compose(transformation_list)

    def __call__(self, img):
        return self.transforms(img)


class ValidTransforms:
    """
    Transforms class for validation dataset
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

    def __call__(self, img):
        return self.transforms(img)
