import torch
from torch import nn


class SymmetricCrossEntropyLoss(nn.Module):
    """
    Acknowledgements:
        https://arxiv.org/pdf/1908.06112
    """

    def __init__(self, alpha: float, beta: float, num_classes: int, cls_weights: torch.Tensor | None = None, reduction: str = "mean", activation_function = nn.Softmax) -> None:
        """
        """
        super(SymmetricCrossEntropyLoss, self).__init__()
        self._alpha = alpha
        self._beta = beta
        self._num_classes = num_classes
        self._cls_weights = cls_weights
        assert reduction in ["mean", "sum"], f"Invalid reduction argument {reduction}"
        self._reduction = torch.mean if "mean" else torch.sum
        self._activation_function = activation_function(dim=1)

    def forward(self, x: torch.Tensor, labels: torch.Tensor, one_hot=True) -> torch.Tensor:
        """
        """
        eps = 1e-7
        x = self._activation_function(x)
        x = torch.clamp(x, min=eps, max=1.0)
        if one_hot:
            y = nn.functional.one_hot(labels, self._num_classes).float()
            y = torch.clamp(y, min=eps, max=1.0)
        else: 
            y = labels
        ce = -torch.sum(torch.log(x) * y, dim=-1)
        rce = -torch.sum(x * torch.log(y), dim=-1)
        if self._cls_weights is None:
            return self._reduction(self._alpha * ce - self._beta * rce)
        cls_weights = torch.Tensor([self._cls_weights[l] for l in labels]).to(x.device)
        return self._reduction(cls_weights * (self._alpha * ce + self._beta * rce))


if __name__ == "__main__":
    raise NotImplementedError("Usage: python main.py args")