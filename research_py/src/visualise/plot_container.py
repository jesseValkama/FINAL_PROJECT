import matplotlib.pyplot as plt
import numpy as np
from src.settings.settings import Settings
from torch.utils.tensorboard import SummaryWriter


class PlotContainer:

    def __init__(self, writer: SummaryWriter, settings: Settings) -> None:
        self._writer = writer
        self._settings = settings
        self._train_graph = np.array([]) 
        self._val_graph = np.array([])
        self._epoch = None
        self._valid_types = ["train", "val"]

    def update_plot(self, value: float, type: str, epoch: int | None = None) -> None:
        assert type in self._valid_types, f"incorrect type, valid types: {self._valid_types}"
        match type:
            case "train":
                self._train_graph = np.append(self._train_graph, value)
            case "val":
                self._val_graph = np.append(self._val_graph, value)
        if epoch is not None:
            self._epoch = epoch

    def push_plots(self) -> None:
        assert self._epoch is not None, "you need to update the plots with update_plot before pushing"
        fig = plt.figure()
        x = np.linspace(1, self._epoch)
        x = np.arange(x)

        plt.plot(x, self._train_graph, c="g", label="Train loss")
        plt.plot(x, self._val_graph, c="b", label="Validation loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.title("Loss curves")
        plt.legend()
        plt.show(fig)
        self._writer.add_figure("Loss curves", fig)


if __name__ == "__main__":
    raise NotImplementedError("Usage: python main.py args")
    