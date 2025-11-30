import cv2 as cv
import numpy as np
from pytorch_grad_cam.utils.image import show_cam_on_image
from src.inference.cams import grad_cam_activations
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
    Function for forward passing with GradCAM
    Args:
        model: the model
        rgb_clip: the rgb clip in LHWC
        post_transforms: normalise the clip with z-scores
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
    grad_cam = grad_cam_activations(acts[0].cpu(), grads[0].cpu())
    acts.clear()
    grads.clear()
    rgb_clip = rgb_clip / 255
    grad_cam = grad_cam.detach().numpy().astype(np.float32) # opencv breaks with fp16
    for i in range(len(grad_cam)):
        heatmap = show_cam_on_image(rgb_clip[i], cv.resize(grad_cam[i], rgb_clip[i][:,:,0].shape), use_rgb=True)
        heatmap = cv.resize(heatmap, (inference_resize, inference_resize)) # cv.putText quality
        heatmap = cv.cvtColor(heatmap, cv.COLOR_RGB2BGR)
        confs = F.softmax(logits, dim=1).view(-1)
        cv.putText(heatmap, f"{dataset_labels[idx]} : {confs[idx].item():.2f}", (30, 40), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2, cv.LINE_AA)
        out.write(heatmap)


@torch.no_grad
def predict_ScoreCAM(model: EfficientLRCN, rgb_clip: np.ndarray, post_transforms: v2.Compose, A_k: List, out: cv.VideoWriter, 
            dataset_labels: List[str], inference_resize: int, BATCH_SIZE: int = 20, dev: str = "cuda:0") -> None:
    """
    Acknowledgements
        https://arxiv.org/pdf/1910.01279 (scorecam official paper)
        https://github.com/jacobgil/pytorch-grad-cam/tree/master (was helpful when trying to understand the paper)
    """
    clip = lhwc2Tensor(rgb_clip, post_transforms).unsqueeze_(0).to(dev) # LHWC -> NLCHW
    with autocast(device_type="cuda"):
        logits = model(clip)
    _, idx = torch.max(logits, dim=1)
    dims = (2,3) # NCHW
    M = F.interpolate(A_k[0], size=rgb_clip.shape[-3:-1], mode="bilinear") # LCHW
    M = s(M, dims)
    M = M[:, :, None, :, :] * clip.squeeze(0)[:, None, :, :, :] # (LCHW -> LC1HW) * (L3HW -> L13HW) -> LC3HW
    M = M.transpose(0, 1).contiguous() # LC3HW -> CL3HW : C works as the batch size
    S_k = torch.Tensor([]).to(dev)
    for i in range(0, M.size(0), BATCH_SIZE):
        batchified = M[i:i+BATCH_SIZE, :, :, :, :]
        outputs = model(batchified)[:, idx]
        S_k = torch.cat((S_k, outputs.view(-1)), dim=0)
    alpha_k = F.softmax(S_k.unsqueeze_(0), dim=1) # C -> 1C
    alpha_k = alpha_k[:, :, None, None]
    score_cam = F.relu(torch.sum(alpha_k.cpu() * A_k[0].cpu(), dim=1))
    A_k.clear()

    rgb_clip = rgb_clip / 255
    score_cam = score_cam.detach().numpy().astype(np.float32) # opencv breaks with fp16
    for i in range(len(score_cam)):
        heatmap = show_cam_on_image(rgb_clip[i], cv.resize(score_cam[i], rgb_clip[i][:,:,0].shape), use_rgb=True)
        heatmap = cv.resize(heatmap, (inference_resize, inference_resize)) # cv.putText quality
        heatmap = cv.cvtColor(heatmap, cv.COLOR_RGB2BGR)
        confs = F.softmax(logits, dim=1).view(-1)
        cv.putText(heatmap, f"{dataset_labels[idx]} : {confs[idx].item():.2f}", (30, 40), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2, cv.LINE_AA)
        out.write(heatmap)


def s(M: torch.Tensor, dims: Tuple) -> torch.Tensor:
    """
    """
    maxs, mins = torch.amax(M, dim=dims), torch.amin(M, dim=dims)
    maxs, mins = maxs[:, :, None, None], mins[:, :, None, None] # NC -> NC11
    return (M - mins) / (maxs - mins + 1e-8)


if __name__ == "__main__":
    raise NotImplementedError("Usage: python main.py")