import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from src.settings.settings import Settings
from torch.utils.tensorboard import SummaryWriter
from typing import List


class PlotContainer:
    """
    Wrapper for the torch SummaryWriter
    Automatically plots graphs for runtime
    and metrics from the testing

    Recommended:
        Run the tensorboard from a separate terminal with:
        tensorboard --logdir=runs
    """

    def __init__(self, writer: SummaryWriter, settings: Settings) -> None:
        """
        """
        self._writer = writer
        self._settings = settings
        self._train_data = {
            "train": {
                "loss": list(),
                "accuracy": list() 
            },
            "val": {
                "loss": list(),
                "accuracy": list() 
            }
        }

    def update_train_plots(self, loss: float, accuracy: float, type: str) -> None:
        """
        """
        if not type in self._train_data:
            raise RuntimeError(f"Incorrect type for updating a plot: {type}")
        self._train_data[type]["loss"].append(loss)
        self._train_data[type]["accuracy"].append(accuracy)
        epoch = len(self._train_data["train"]["loss"])
        self._writer.add_scalar(type + " loss", loss, global_step=epoch)
        self._writer.add_scalar(type + " accuracy", accuracy, global_step=epoch)
                
    def push_train_plots(self) -> None:
        """
        """
        epoch = len(self._train_data["train"]["loss"])
        assert epoch > 0, "you need to update the plots with update_plot before pushing"
        fig, axs = plt.subplots(2, 1)
        fig.suptitle("Training metrics")
        plt.subplots_adjust(hspace=0.5)

        x = np.linspace(1, epoch, epoch)
        x_val = np.arange(self._settings.validation_interval, epoch + 1, self._settings.validation_interval)
        train_loss = np.array(self._train_data["train"]["loss"])
        val_loss = np.array(self._train_data["val"]["loss"])
        train_accuracy = np.array(self._train_data["train"]["accuracy"])
        val_accuracy = np.array(self._train_data["val"]["accuracy"])
        val_loss_interp = np.interp(x, x_val, val_loss)
        val_accuracy_interp = np.interp(x, x_val, val_accuracy)

        axs[0].plot(x, train_loss, c="g", label="Train loss")
        axs[0].plot(x, val_loss_interp, c="b", label="Validation loss")
        axs[0].scatter(x, train_loss, c="g")
        axs[0].scatter(x_val, val_loss, c="b")
        axs[0].set_ylabel("Loss")
        axs[0].set_xlabel("Epoch")
        axs[0].set_title("Loss curves")
        axs[0].legend()

        axs[1].plot(x, train_accuracy, c="g", label="Train accuracy")
        axs[1].plot(x, val_accuracy_interp, c="b", label="Validation accuracy")
        axs[1].scatter(x, train_accuracy, c="g")
        axs[1].scatter(x_val, val_accuracy, c="b")
        axs[1].set_ylabel("Accuracy")
        axs[1].set_xlabel("Epoch")
        axs[1].set_title("Accuracy curves")
        axs[1].legend()

        self._writer.add_figure("Training metrics", fig)
    
    def push_conf_mat(self, cm: np.ndarray, labels: List[str]) -> None:
        """
        """
        df_cm = pd.DataFrame(cm, index=[l for l in labels], columns=[l for l in labels])
        fig = plt.figure()
        sns.heatmap(df_cm, annot=True)
        self._writer.add_figure("Confusion Matrix", fig)
    
    def push_tsne(self, latent_repr: np.ndarray) -> None:
        """
        """
        fig = plt.figure()
        plt.scatter(latent_repr[:, 0], latent_repr[:, 1])
        self._writer.add_figure("Embeddings in the latent space", fig)
    

if __name__ == "__main__":
    raise NotImplementedError("Usage: python main.py args")
    