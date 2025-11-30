import cv2 as cv
import numpy as np
import torch
from torchvision.transforms import v2


def remove_black_borders(img: np.ndarray, black_thresh: int = 5) -> np.ndarray:
    """
    Removes the black borders from a given image, e.g., if a recorded on a phone
    Args:
        img: the input image
        black_thresh: the lower threshold as some of the times the borders might not be as 0
    Returns:
        np.ndarray: the img without black borders
    """
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(grey, black_thresh, 255, cv.THRESH_BINARY)
    coords = cv.findNonZero(thresh)
    x, y, w, h = cv.boundingRect(coords) # xy = top left not centre
    return img[y:y+h, x:x+w]


def pad2square(img: np.ndarray, resize: int | None = None) -> np.ndarray:
    """
    Pads the image to be a square for normalisation, necessary as OOPS has varying aspect ratios
    Args:
        img: the input image
        resize: optional to resize the image
    Retuns:
        np.ndarray: the padded image
    """
    h, w = img.shape[:-1]
    pad_y = max(0, (w - h) // 2)
    pad_x = max(0, (h - w) // 2)
    img = cv.copyMakeBorder(img, pad_y,  pad_y, pad_x, pad_x, cv.BORDER_CONSTANT)
    if resize is not None:
        img = cv.resize(img, (resize, resize))
    return img

def lhwc2Tensor(clip: np.ndarray, post_transforms: v2.Compose | None = None, dev: str | None = None) -> torch.Tensor:
    """
    Function to conver clip of np.array to Tensor given format lhwc
    Args:
        clip: the input clip
        post_transforms: optional e.g., z-scale
        dev: optional, move to dev
    Returns:
        torch.Tensor: the output clip
    """
    clip = torch.Tensor(clip).permute([0, 3, 1, 2]).contiguous().to(torch.float)
    clip.div_(255.0)
    if post_transforms:
        clip = post_transforms(clip)
    if dev:
        clip = clip.to(dev)
    return clip



if __name__ == "__main__":
    raise NotImplementedError("Usage: python main.py args")