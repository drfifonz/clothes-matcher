import torch.nn as nn


class BaseModel(nn.Module):
    """
    Base model class used to inheritance by specific ones
    """

    def __init__(self) -> None:
        super().__init__()
