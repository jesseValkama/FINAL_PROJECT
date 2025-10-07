import numpy as np


def get_weight_INS(samples, ignore_thresh = 10, weight_factor = 1):
    """
    Caculates the inverse of number of samples (INS):

    acknowledgements:
        https://medium.com/gumgum-tech/handling-class-imbalance-by-introducing-sample-weighting-in-the-loss-function-3bdebd8203b4

    Args:
        samples: the n of samples per class
        power: the factor of how much to scale the weights (0 == all 1 unless 0 samples for a class)

    Returns:
        the updated weights

    """

    assert 0 <= weight_factor <= 1, "Use a power between 0 and 1"
    n = len(samples)

    # avoids 0 div error
    weights = np.maximum(samples, 1)
    weights = 1.0 / np.array(np.power(weights, weight_factor))
    weights = weights / np.sum(weights) * n
    weights[samples < ignore_thresh] = 0.0
    return weights
