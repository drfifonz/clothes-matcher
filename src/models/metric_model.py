import torch.nn as nn
from pytorch_metric_learning.utils import common_functions


class MetricModel(nn.Module):
    """
    metric model class
    """

    def __init__(self, base_model: nn.Module) -> None:
        super().__init__()
        self.base_model = base_model
        self.base_model.fc = common_functions.Identity()

    def forward(self, data):
        """
        forward funcion
        """
        # TODO forward function
        return self.base_model(data)