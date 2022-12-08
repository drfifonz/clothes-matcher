import matplotlib.pyplot as plt


class Visualizer:
    """
    Class for visualization learing effects with plots
    """

    def __init__(self, size: tuple = (8, 6)) -> None:
        self.figure, self.axes = plt.subplots()
        self.figure.set_size_inches(size)
        self.axes.grid(True)
        self.title = "Loss accuracy on Dataset"

        self.axes.set_title(self.title)

    def plot_data(self, x_data: list[int], y_data: list[float]) -> None:
        """
        Plots data
        """
        self.axes.plot(x_data, y_data)

    def change_title(self, title: str) -> None:
        """
        change title
        """
        self.title = title
        self.axes.set_title(self.title)

    def show(self) -> None:
        """
        display plots
        """
        plt.show()

    def save_plot(self, path: str) -> None:
        """
        Save plot as a png file at specified path.
        """
        pass
