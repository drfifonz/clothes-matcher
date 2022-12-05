import torch.nn as nn


class BboxDetectionModel(nn.Module):
    """
    Bbox detection model class
    """

    # TODO Bbox detection Model class needs to be refilled
    def __init__(self, base_model, num_labels: int) -> None:
        super().__init__()
        self.base_model = base_model
        self.num_labels = num_labels

        # TODO consider other name for regressor
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

        # TODO consider other name for classifier
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
