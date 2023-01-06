import torch.nn as nn


class BboxDetectionModel(nn.Module):
    """
    Bbox detection model class
    """

    def __init__(self, base_model, num_labels: int) -> None:
        super().__init__()
        self.base_model = base_model
        self.num_labels = num_labels

        # regressor is user for bounding box position
        self.regressor = nn.Sequential(
            nn.Linear(base_model.fc.in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
            nn.Sigmoid(),
        )

        # create output from convolution block
        self.classifier = nn.Sequential(
            nn.Linear(base_model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, self.num_labels),
        )

        self.base_model.fc = nn.Identity()

    def forward(self, data):
        """
        pass inputs through base model
        """
        features = self.base_model(data)
        bboxes = self.regressor(features)
        classifier = self.classifier(features)

        return (bboxes, classifier)
