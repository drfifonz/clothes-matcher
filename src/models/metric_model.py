import torch.nn as nn


class MetricModel(nn.Module):
    """
    metric model class
    """

    def __init__(self, base_model: nn.Module) -> None:
        super().__init__()
        self.base_model = base_model

    def forward(self, data):
        """
        forward funcion
        """
        # TODO forward function
        return data
