import torch
import torch.nn.functional as F


def grad_cam_activations(acts: torch.Tensor, grads: torch.Tensor) -> torch.Tensor:
    """
    function to calculate the gradcam localisation map
    math explanation in the official grad-cam paper
        https://arxiv.org/pdf/1610.02391
    Args:
        acts: the activation from the forward pass
        grads: the grads for the logit w.r.t to the activations
    Returns:
        the localisation map
    """
    alpha = torch.mean(grads, dim=(2,3)) # NCHW -> NC (gap)
    alpha.unsqueeze_(2).unsqueeze_(2)
    return F.relu(torch.sum(acts * alpha, dim=1)) # NCHW * NC11 -> NHW (linear combination)


if __name__ == "__main__":
    raise NotImplementedError("Usage: python main.py args")