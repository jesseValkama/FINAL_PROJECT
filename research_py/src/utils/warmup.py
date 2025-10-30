from torch import optim
from torch.optim.lr_scheduler import LRScheduler
from typing import List
from typing_extensions import override


class WarmupLR(LRScheduler):
    """
    """

    def __init__(self, optimiser: optim.Optimizer, epochs: int, total_iters: int, last_iter: int = -1) -> None:
        """
        """
        super().__init__(optimiser, last_iter)
        assert last_iter < 0, "last_iter should be negative: current_iter += -1 * last_iter"
        self._learning_rates = [group["lr"] for group in optimiser.param_groups]
        self._epochs = epochs
        self._total_iters = total_iters 
        self._last_iter = last_iter 
        self._current_iter = 1

    @override
    def get_lr(self) -> List[float]:
        """
        """
        if self._is_initial:
            return [group["lr"] for group in self.optimizer.param_groups]

        learning_rates = [self._learning_rates[i] * self._current_iter / self._total_iters for i, _ in enumerate(self.optimizer.param_groups)]
        self._current_iter += -1 * self._last_iter
        return learning_rates
    

if __name__ == "__main__":
    raise NotImplementedError("Usage: python main.py args")