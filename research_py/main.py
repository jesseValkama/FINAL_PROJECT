import os
from pathlib import Path
from src.inference.inference import run_inference
from src.train.train_loop import run_loop 
from src.settings import Settings


def main() -> None:
    """
    This is the main file, run everything from here
    Uses both command-line arguments and also a settings.yaml

    Command-line args:
        train: 0 | 1
        test: 0 | 1
        inference: 0 | 1
    """
    settings = Settings()
    project_dir = Path(settings.project_dir)
    assert project_dir.exists, "Enter a valid project directory, no need for main.py"
    os.chdir(project_dir)
    # scorecam_test()
    if settings.train or settings.test:
        run_loop(settings=settings)
    if settings.inference:
        run_inference(settings, "ScoreCAM")


def scorecam_test():
    from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from torchvision.models import resnet50, ResNet50_Weights
    import cv2 as cv
    import torch

    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    target_layers = [model.layer4[-1]]
    rgb_img = cv.imread("C:/Datasets/dog_cat.jfif") / 255
    input_tensor = torch.Tensor(rgb_img).permute([2, 0, 1]).contiguous().unsqueeze_(0)
    targets = [ClassifierOutputTarget(281)]
    with ScoreCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        model_outputs = cam.outputs


if __name__ == "__main__":
    main()