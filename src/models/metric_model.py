import torch.nn as nn

# from pytorch_metric_learning.utils import common_functions


class MetricModel(nn.Module):
    """
    metric model class
    """

    def __init__(self, base_model: nn.Module, embedding_size: int = 100) -> None:
        super().__init__()
        self.base_model = base_model
        # self.base_model.fc = common_functions.Identity()
        self.base_model.fc = nn.Linear(in_features=base_model.fc.in_features, out_features=embedding_size, bias=True)

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


class Embedder(nn.Module):
    """
    Simple embeddings model.
    """

    def __init__(self, layer_sizes, final_relu=False) -> None:
        super().__init__()
        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=False))
            layer_list.append(nn.Linear(input_size, curr_size))
        self.net = nn.Sequential(*layer_list)
        self.last_linear = self.net[-1]

    def forward(self, data):
        """
        forward function
        """
        return self.net(data)
