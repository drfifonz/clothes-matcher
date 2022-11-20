import torch.nn as nn


class BboxDetectionModel(nn.Module):
    """
    Bbox detection model class
    """

    def __init__(self, base_model, num_labels: int) -> None:
        super().__init__()
        self.base_model = base_model
        self.num_labels = num_labels

        # TODO consider better name for regressor
        # regressor is user for bounding box position
        self.regressor = nn.Sequential(
            nn.Linear(base_model.fc.in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
            nn.ReLU(),
            nn.Sigmoid(),
        )

        # TODO implement classifier
        # create output from convolution block
        self.classifier = nn.Sequential()

        self.base_model.fc = nn.Identity()

    def forward(self, x):
        features = self.base_model(x)
        bboxes = self.regressor(features)
        classifier = self.classifier(features)

        return (bboxes, classifier)
