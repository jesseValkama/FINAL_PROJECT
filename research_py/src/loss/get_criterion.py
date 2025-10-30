from src.loss.symmetric_cross_entropy_loss import SymmetricCrossEntropyLoss
from src.settings.settings import Settings
import torch
from torch import nn


def get_criterion(criterion: str, cls_weights: torch.Tensor, settings: Settings):
    """
    """
    match criterion:
        case "ce":
            return nn.CrossEntropyLoss(weight=cls_weights, label_smoothing=settings.label_smoothing)
        case "sce":
            return SymmetricCrossEntropyLoss(settings.sce_alpha, settings.sce_beta, num_classes=len(settings.dataset_labels), cls_weights=cls_weights)


if __name__ == "__main__":
    raise NotImplementedError("Usage: python main.py args")