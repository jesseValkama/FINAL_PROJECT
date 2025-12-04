import numpy as np
from src.settings.settings import Settings
import torch


def get_cls_weights(samples: np.ndarray, settings: Settings, print_weights: bool = True) -> torch.Tensor:
    """
    Calcualates the ins and normalises the calculated classes to sum to the len of the weights
    Also applies the pre-determined weights from the settings
    Args:
        samples: array of samples per cls
        settings: settings
    Returns:
        torch.Tensor: the new balanced weights on the training dev
    """
    pre_weights = get_weight_INS(samples, settings.cls_ignore_thresh, settings.cls_weight_factor)
    post_weights = np.array(settings.label_weights)
    weights = pre_weights * post_weights
    sum = np.sum(weights)
    total = len([x for x in weights if x > 0.0])
    weights *= total / sum
    if print_weights:
        print("The weights are:")
        for i in range(len(settings.dataset_labels)):
            print(f"  {settings.dataset_labels[i]}: {weights[i]}")
    return torch.Tensor(weights).to(settings.train_dev)


def get_weight_INS(samples: np.ndarray, ignore_thresh: int = 10, weight_factor: int = 1) -> np.ndarray:
    """
    Caculates the inverse of number of samples (INS):
    acknowledgements:
        https://medium.com/gumgum-tech/handling-class-imbalance-by-introducing-sample-weighting-in-the-loss-function-3bdebd8203b4
    Args:
        samples: the n of samples per class
        ignore_thresh: set a class weight to 0.0 if there are < thresh samples
        power: the factor of how much to scale the weights (0 == all 1 unless 0 samples for a class)
    Returns:
        np.ndarray: the updated weights
    """
    assert 0 <= weight_factor <= 1, "Use a power between 0 and 1"
    n = len(samples)
    weights = np.maximum(samples, 1) # avoids 0 div error
    weights = 1.0 / np.array(np.power(weights, weight_factor))
    weights = weights / np.sum(weights) * n
    weights[samples < ignore_thresh] = 0.0
    return weights


if __name__ == "__main__":
    raise NotImplementedError("Usage: python main.py args")