from src.models.efficientnet_lrcn import EfficientLRCN
from src.settings.settings import Settings
import torch
from torch import nn
import torch.nn.functional as F


class TRADES(nn.Module):
    """
    Acknowledgements:
        https://arxiv.org/pdf/1901.08573 (official paper)
        https://github.com/yaodongyu/TRADES (official repo)
        This code is untested due to cuda out of memory
    """

    def __init__(self, settings: Settings) -> None:
        """
        """
        super(TRADES, self).__init__()
        self._adversarial_factor = settings.trades_adversarial_factor
        self._trades_steps = settings.trades_steps
        self._step_size = settings.trades_step_size
        self._kl_div = nn.KLDivLoss(reduction="sum")
        self._batch_size = settings.train_batch_size
        self._eps = settings.trades_epsilon
        self._beta = settings.trades_beta

    def forward(self, model: EfficientLRCN, vids: torch.Tensor) -> torch.Tensor:
        """
        """
        model.eval()
        adversarial_examples = vids.detach() * (self._adversarial_factor * torch.rand(vids.shape)).to(vids.device).detach()
        for _ in range(self._trades_steps):
            adversarial_examples.requires_grad_()
            with torch.enable_grad():
                kl_loss = self._kl_div(F.log_softmax(model(adversarial_examples), dim=1), F.softmax(model(vids), dim=1)) 
            grad = torch.autograd.grad(kl_loss, [adversarial_examples])[0]
            adversarial_examples = adversarial_examples.detach() + self._step_size * torch.sign(grad.detach())
            adversarial_examples = torch.min(torch.max(adversarial_examples, vids - self._eps), vids + self._eps)
            torch.clamp_(adversarial_examples, 0.0, 1.0)
        model.train()
        return torch.clamp_(adversarial_examples, 0.0, 1.0).requireds_grad_(False)
    
    def calc_loss(self, model: EfficientLRCN, vids: torch.Tensor, natural_loss: torch.Tensor, adversarial_examples: torch.Tensor):
        """
        """
        robust_loss = (1.0 / self._batch_size) * self._kl_div(F.log_softmax(model(adversarial_examples), dim=1), F.softmax(model(vids), dim=1))
        return natural_loss + self._beta * robust_loss


if __name__ == "__main__":
    raise NotImplementedError("Usage: pyhon main.py args")