import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from src.inference.predict import predict_GradCAM, predict_ScoreCAM
from src.models.efficientnet_lrcn import EfficientLRCN
from src.settings.settings import Settings
from src.utils.preprocess import pad2square, remove_black_borders
import torch
from torchvision.transforms import v2


def run_inference(settings: Settings, cam_name: str = "ScoreCAM", capture_interval: int = 10) -> None:
    """
    acknowledgements:
        https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html (video reading and writing)
    The function for running "inference" as in pre-recorded videos
    Args:
        settings: the settings
        cam_name: the XAI mode to use either GradCAM or ScoreCAM
        capture_interval: every how many frames to use for the clip that is forward passed
    """
    print("Starting inference")
    assert cam_name in ["ScoreCAM", "GradCAM"], f"Enter a supported cam: {cam_name}"
    model_name = settings.work_model if settings.train else settings.inference_model
    save_path = os.path.join(settings.weights_path, model_name + ".pth")
    model = EfficientLRCN(settings)
    model.load_state_dict(torch.load(save_path))
    model.to(settings.train_dev)
    model.eval()
    model.rnn.train() # pytorch crashes otherwise
    acts = list()
    grads = list()
    forward_hook = model.point_wise.register_forward_hook(
        lambda module, input, output : acts.append(output))
    if cam_name == "GradCAM":
        backward_hook = model.point_wise.register_full_backward_hook(
            lambda module, input, output : grads.append(output[0]))
    assert Path(settings.inference_path).is_dir(), "Enter a proper inference dir (connect the external ssd)"
    video_paths = os.listdir(Path(settings.inference_path))
    post_transforms = v2.Compose([v2.Normalize(mean=settings.mean, std=settings.standard_deviation, inplace=True)])
    save_dir = settings.inference_save_dir
    assert Path(save_dir).is_dir(), "set a proper inference save dir"

    for video_path in video_paths:
        cap = cv.VideoCapture(os.path.join(settings.inference_path, video_path))
        fourcc = cv.VideoWriter_fourcc(*'XVID') # todo: fix the path is not .original.avi e.g., .mp4.avi
        out = cv.VideoWriter(os.path.join(save_dir, "saved_" + video_path + "_" + cam_name + ".avi"), fourcc, 
                             float(cap.get(cv.CAP_PROP_FPS)/capture_interval), (settings.inference_save_res,  settings.inference_save_res))
        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        clip = list()
        frame_idx = 0
        vid_idx = 0
        while True:
            ret, img = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            frame_idx += 1
            if frame_idx % capture_interval == 0:
                img = remove_black_borders(img)
                img = pad2square(img, settings.image_size)
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                clip.append(img)
                if len(clip) >= settings.video_length:
                    vid_idx += 1
                    match cam_name:
                        case "GradCAM":
                            predict_GradCAM(model, np.array(clip), post_transforms, acts, grads, 
                                            out, settings.dataset_labels, settings.inference_save_res, dev=settings.train_dev)
                        case "ScoreCAM":
                            predict_ScoreCAM(model, np.array(clip), post_transforms, acts, 
                                             out, settings.dataset_labels, settings.inference_save_res, dev=settings.train_dev)
                    clip.clear()
        cap.release()
    forward_hook.remove()
    if cam_name == "GradCAM":
        backward_hook.remove()


if __name__ == "__main__":
    raise NotImplementedError("Usage: python main.py args")