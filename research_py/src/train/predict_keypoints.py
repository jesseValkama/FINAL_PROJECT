import cv2 as cv
import numpy as np
import os
from pathlib import Path
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from src import Settings
import torch
from typing import List, Optional
from ultralytics import YOLO


class Target:
    def __init__(self, category: int) -> None:
        self.category: int = category

    def __call__(self, model_output) -> torch.Tensor:
        model_output = model_output[0]
        if len(model_output.shape) == 1:
            return model_output[4 + self.category]
        return model_output[:, 4 + self.category]


def kp_yolo_gradcam() -> None:
    """
    currently doesn't work because the ultralytics pipeline
    is way too fancy -> i need to make a yolo implementation
    or a wrapper that fixes this stuff

    the problem is that yolo doesn't retain grad in forward pass
    -> backprop won't work
    """
    model = YOLO("research_py/weights/ultralytics/yolo11s.pt").model
    target_layers: List = [model.model[7].conv]
    targets: List = [Target(0)]

    img_path = Path("D:/datasets/cifar10_inference_imgs/inf0.jpg")
    assert(img_path.exists())
    input_img: Optional[np.ndarray] = cv.imread(str(img_path), cv.IMREAD_COLOR)
    assert(input_img is not None)
    timg: torch.Tensor = torch.tensor(input_img)
    timg = timg.permute([2,0,1]).to(torch.float)
    timg.unsqueeze_(0).div_(255.0)

    with GradCAM(model=model, target_layers=target_layers) as cam:
        greyscale_cam: np.ndarray = cam(input_tensor=timg, targets=targets)
        greyscale_cam = greyscale_cam[0, :]
        visualisation: np.ndarray = show_cam_on_image(input_img, greyscale_cam, use_rgb=False)
        cv.imshow("cam", visualisation)
        cv.waitKey(0)
        cv.destroyAllWindows()


def kp_yolo(model, videos: torch.Tensor, settings: Settings) -> torch.Tensor:
    """
    Function to get keypoints from yolo

    TODO: fix hack fixes

    Args:

    Returns:

    """
    keypoints = torch.Tensor([]).to(settings.train_dev)
    for frame in range(videos.shape[1]):
        results = model.predict(videos[:,frame,:,:,:], verbose=False, imgsz=settings.image_size, device=settings.train_dev, half=True)
        batch_wise = torch.Tensor([]).to(settings.train_dev)
        for result in results:
            kps = result.keypoints.xyn
            if kps.shape[0] != 0:
                # > 1 detections mess up the entire pipeline: all datasets contain a single person, 1 detection = failed nms
                kps = kps[0] if kps.shape[0] > 1 else kps # hack fix even for me
                batch_wise = torch.cat((batch_wise, kps.view(1, -1)), dim=0)
            else:
                # adding a tensor of zeros for if no people / detections to keep lengths the same
                batch_wise = torch.cat((batch_wise, torch.zeros((1, kps.shape[-2] * kps.shape[-1]), device=settings.train_dev)), dim=0)
        batch_wise.unsqueeze_(1) # (bs, xy) -> (bs, seq, xy)
        keypoints = torch.cat((keypoints, batch_wise), dim=1)
    return (keypoints - 0.5) * 2


if __name__ == "__main__":
    raise NotImplementedError("Usage: python main.py args")