import cv2 as cv
import numpy as np
from src.inference.write import write_video
from src.models.efficientnet_lrcn import EfficientLRCN
from src.utils.preprocess import lhwc2Tensor
import torch
from torch.amp import autocast
import torch.nn.functional as F
from torchvision.transforms import v2
from typing import List, Tuple


def predict_GradCAM(model: EfficientLRCN, rgb_clip: np.ndarray, post_transforms: v2.Compose, acts: List, grads: List, out: cv.VideoWriter, 
            dataset_labels: List[str], inference_resize: int, dev: str = "cuda:0") -> None:
    """
    GradCAM modified to work with vidoe data
    Acknowledgements:
        https://arxiv.org/pdf/1610.02391
    Args:
        model: the model
        rgb_clip: the rgb clip in LHWC
        post_transforms: normalise the clip with z-scaling
        acts: list to store activations
        grads: list to store gradients
        out: the video writer
        dataset_labels: list of labels as str
        inference_resize: the output size for the written vid
        dev: inference device
    """
    clip = lhwc2Tensor(rgb_clip, post_transforms).unsqueeze_(0).to(dev) # LCHW -> NLCHW
    with autocast(device_type="cuda"):
        logits = model(clip)
    logit, idx = torch.max(logits, dim=1)
    logit.backward()
    alpha = torch.mean(grads[0], dim=(2,3))
    alpha = alpha[:, :, None, None]
    grad_cam = F.relu(torch.sum(acts[0].cpu() * alpha.cpu(), dim=1))
    acts.clear()
    grads.clear()
    write_video(rgb_clip, grad_cam, out, logits, idx, inference_resize, dataset_labels)
    

@torch.no_grad
def predict_ScoreCAM(model: EfficientLRCN, rgb_clip: np.ndarray, post_transforms: v2.Compose, A_k: List, out: cv.VideoWriter, 
            dataset_labels: List[str], inference_resize: int, BATCH_SIZE: int = 20, dev: str = "cuda:0") -> None:
    """
    ScoreCAM modified to work with video data (could be bugged as the activations are spiral shaped)
    Acknowledgements
        https://arxiv.org/pdf/1910.01279 (scorecam official paper)
        https://github.com/jacobgil/pytorch-grad-cam/tree/master (was helpful when trying to understand the paper)
        https://github.com/haofanwang/Score-CAM/blob/master/cam/scorecam.py#L52 (official impl)
    Args:
        model: the model
        rgb_clip: the rgb clip in LHWC
        post_transforms: normalise the clip with z-scaling
        A_k: list to store activations
        out: the video writer
        dataset_labels: list of labels as str
        inference_resize: the output size for the written vid
        batch_size: batch_size for batching the channels for scorecam
        dev: inference device 
    """
    clip = lhwc2Tensor(rgb_clip, post_transforms).unsqueeze_(0).to(dev) # LHWC -> NLCHW
    with autocast(device_type="cuda"):
        logits = model(clip)
    _, idx = torch.max(logits, dim=1)
    up = F.interpolate(A_k[0], size=rgb_clip.shape[-3:-1], mode="bilinear")
    M = s(up, dims=(2,3)) # NCHW
    M = M[:, :, None, :, :] * clip.squeeze(0)[:, None, :, :, :] # (LCHW -> LC1HW) * (L3HW -> L13HW) -> LC3HW
    M = M.transpose(0, 1).contiguous() # LC3HW -> CL3HW : C works as the batch size
    S_k = torch.Tensor([]).to(dev)
    for i in range(0, M.size(0), BATCH_SIZE):
        batchified = M[i:i+BATCH_SIZE, :, :, :, :] # am i reading harry potter or a research paper
        outputs = model(batchified)[:, idx]
        S_k = torch.cat((S_k, outputs.view(-1)), dim=0)
    alpha_k = F.softmax(S_k.unsqueeze_(0), dim=1) # C -> 1C, no need for normalisation as the official impl doesn't use it
    alpha_k = alpha_k[:, :, None, None] # 1C11
    score_cam = F.relu(torch.sum(alpha_k.cpu() * up.cpu(), dim=1)) # official impl uses upscaled instead of A_k
    A_k.clear()
    write_video(rgb_clip, score_cam, out, logits, idx, inference_resize, dataset_labels)
    

def s(M: torch.Tensor, dims: Tuple[int, int]) -> torch.Tensor:
    """
    Normalisation function for scorecam:
        https://arxiv.org/pdf/1910.01279
    Args:
        M: the upsampled activations
        dims: the dims to take the max and min over
    Returns:
        torch.Tensor: the normalised upsampled activations
    """
    maxs, mins = torch.amax(M, dim=dims), torch.amin(M, dim=dims)
    maxs, mins = maxs[:, :, None, None], mins[:, :, None, None] # NC -> NC11
    return (M - mins) / (maxs - mins + 1e-8)


if __name__ == "__main__":
    raise NotImplementedError("Usage: python main.py")