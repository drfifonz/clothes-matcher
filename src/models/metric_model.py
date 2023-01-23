import torch.nn as nn

from pytorch_metric_learning.utils import common_functions


class MetricModel(nn.Module):
    """
    metric model class
    """

    def __init__(self, base_model: nn.Module, embedding_size: int = 64) -> None:
        super().__init__()
        self.base_model = base_model
        self.base_model.fc = common_functions.Identity()
        # self.embedder = nn.Linear(in_features=base_model.fc.in_features, out_features=embedding_size, bias=True)

    def forward(self, data):
        """
        forward funcion
        """
        return self.base_model(data)

    # def forward(self, data):
    #     """
    #     forward funcion
    #     """
    #     features = self.base_model(data)

    #     return self.embedder(features)
