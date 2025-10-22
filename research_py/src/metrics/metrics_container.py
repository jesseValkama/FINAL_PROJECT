import numpy as np
from src.visualise.plot_container import PlotContainer
import torch
from torch import nn
from typing import List


class MetricsContainer:
    """
    """

    def __init__(self, dataset_labels: List[str], plot_container: PlotContainer, activation_function = nn.Softmax) -> None:
        """
        """
        self._plot_container = plot_container
        self._dataset_labels = dataset_labels
        self._activation_function = activation_function(dim=1)
        n = len(dataset_labels)
        self._conf_mat_elements = {dataset_labels[i]: {"tp": 0, "fp": 0, "fn": 0} for i in range(n)}
        self._conf_mat_table = np.zeros((n, n))
        self._preds_total = np.zeros(n)
        self._preds_correct = np.zeros(n)
        self._10_class = None
        self._fall = None
        self._fallen = None 
        self._fall_U_fallen = None
        self._recall = None
        self._precision = None
        self._f1 = None
        
    def calc_iter(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        """
        """
        candidates = self._activation_function(logits)
        candidates, labels = candidates.detach().cpu().numpy(), labels.detach().cpu().numpy()
        preds = np.argmax(candidates, axis=1)
        
        n = len(preds)
        for i in range(n):
            self._conf_mat_table[labels[i]][preds[i]] += 1
            self._preds_total[labels[i]] += 1
            if preds[i] == labels[i]:
                self._conf_mat_elements[self._dataset_labels[preds[i]]]["tp"] += 1
                self._preds_correct[labels[i]] += 1
                continue
            self._conf_mat_elements[self._dataset_labels[preds[i]]]["fp"] += 1
            self._conf_mat_elements[self._dataset_labels[labels[i]]]["fn"] += 1

    def calc_metrics(self) -> None:
        """
        """
        classes = self._conf_mat_elements.keys()
        self._recall = np.array([ 
            self._conf_mat_elements[cls]["tp"]/(self._conf_mat_elements[cls]["tp"] + self._conf_mat_elements[cls]["fn"])
            if self._conf_mat_elements[cls]["tp"] + self._conf_mat_elements[cls]["fn"] > 0 else 0.0
            for cls in classes 
        ])
        self._precision = np.array([
            self._conf_mat_elements[cls]["tp"]/(self._conf_mat_elements[cls]["tp"] + self._conf_mat_elements[cls]["fp"]) 
            if self._conf_mat_elements[cls]["tp"] + self._conf_mat_elements[cls]["fp"] > 0 else 0.0
            for cls in classes
        ])
        self._f1 = np.array([
            2 * (self._precision[cls] * self._recall[cls]) / (self._precision[cls] + self._recall[cls])
            if self._precision[cls] + self._recall[cls] > 0 else 0.0
            for cls in range(len(classes)) 
        ])
        self._fall = {"recall": self._recall[1], "precision": self._precision[1], "f1": self._f1[1]}
        self._fallen = {"recall": self._recall[2], "precision": self._precision[2], "f1": self._f1[2]}
        self._fall_U_fallen = {"recall": np.mean(self._recall[1:3]), "precision": np.mean(self._precision[1:3]), "f1": np.mean(self._f1[1:3])}
        self._10_class = {"balanced_accuracy": (np.nanmean(self._precision) + np.nanmean(self._recall)) / 2, "accuracy": np.sum(self._preds_correct) / np.sum(self._preds_total), "f1": np.nanmean(self._f1)}
        
    def show_conf_mat(self) -> None:
        """
        """
        ...

    def show_metrics(self) -> None:
        """
        """
        ...

    def print_conf_mat(self) -> None:
        """
        """
        assert np.sum(self._conf_mat_table) != 0.0, "Construct the confusion matrix first with calc_iter method"
        assert np.sum(self._conf_mat_table) == np.sum(self._preds_total), \
        f"conf mat messed up: conf mat {np.sum(self._conf_mat_table)}, total {np.sum(self._preds_total)}"
        print("---------------START OF THE CONFUSION MATRIX---------------\n\n")
        print(self._conf_mat_table)
        print("\n\n----------------END OF THE CONFUSION MATRIX----------------")
        print("\n\n")

    def print_metrics(self) -> None:
        """
        """
        assert self._10_class is not None, "Calculate the metrics before printing"
        n = len(self._dataset_labels)
        print("-------------------START OF THE METRICS-------------------\n\n")
        for i in range(n):
            print(f"{self._dataset_labels[i]}:")
            print(f"Recall: {self._recall[i]:.2f}")
            print(f"Precision: {self._precision[i]:.2f}")
            print(f"F1: {self._f1[i]:.2f}\n")
        print("\n--------------------END OF THE METRICS--------------------")
        print("\n\n")


if __name__ == "__main__":
    raise NotImplementedError("Usage: python main.py args")