import matplotlib.pylab as plt
from PIL import Image
import numpy as np
import utils.config as cfg
from datasets import LandmarkHDF5Dataset
from utils.models_utils import TrainTransforms


class Visualize:
    """
    class for visualization effect
    """

    IMAGE_SIZE = 128

    def __init__(
        self,
        val_dataset,
        test_dataset,
    ) -> None:

        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def predictions_comparison(self, distance):
        """
        Comparison of random Test & Val model
        """
        plt.style.use("ggplot")
        plt.figure(figsize=(20, 11))
        for i, idx in enumerate(np.random.choice(len(self.val_dataset), size=25, replace=False)):
            matched_idx = distance[idx].argmin().item()

            actual_label = self.test_dataset.labels[idx]
            predicted_label = self.val_dataset.labels[matched_idx]

            actual_image = self.test_dataset.images[idx]
            predicted_image = self.val_dataset.images[matched_idx]

            actual_image = np.array(Image.fromarray(actual_image).resize((self.IMAGE_SIZE, self.IMAGE_SIZE)))
            predicted_image = np.array(Image.fromarray(predicted_image).resize((self.IMAGE_SIZE, self.IMAGE_SIZE)))

            stack = np.hstack([actual_image, predicted_image])

            plt.subplot(5, 5, i + 1)
            plt.imshow(stack)
            plt.title(f"Get: {actual_label}\nPredicted: {predicted_label}", fontdict={"fontsize": 8})
            plt.axis("off")

            plt.show()


if __name__ == "__main__":
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    tf_list = TrainTransforms(mean=MEAN, std=STD, as_hdf5=True)

    test_ds = LandmarkHDF5Dataset(
        root=cfg.HDF5_DIR_PATH,
        running_mode="test",
        transforms_list=tf_list,
        measure_time=False,
        is_metric=True,
    )
    val_ds = LandmarkHDF5Dataset(
        root=cfg.HDF5_DIR_PATH,
        running_mode="val",
        transforms_list=tf_list,
        measure_time=False,
        is_metric=True,
    )

    visualizer = Visualize(
        test_dataset=test_ds,
        val_dataset=val_ds,
    )
    # TODO add distance to visualizer
    visualizer.predictions_comparison(distance="")
