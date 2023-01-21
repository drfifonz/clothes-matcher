import logging
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

class TrainUtils:
    """
    Utils class for metric model training
    """
    def __init__(self) -> None:
        pass

    def get_visualizer_hook(self, umapper, umap_embeddings, labels, split, keyname, *args):
        """
        Returns a plot of embeddings using umap 
        https://umap-learn.readthedocs.io/en/latest/
        """
        logging.info(f"UMAP Plot for split: {split}")
        labels_unique = np.unique(labels)
        num_classes = len(labels_unique)

        plt.figure(figsize=(20, 15))
        plt.gca().set_prop_cycle(
            cycler(
                "color", [plt.cm.nipy_spectral(i) for i in np.linspace(0, 0.9, num_classes)]
            ) 
        )

        for i in range(num_classes):
            index = labels == labels_unique[i]
            plt.plot(umap_embeddings[index, 0], umap_embeddings[index, 1], ".", markersize=1)
        plt.show()

