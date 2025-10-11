from src.settings.settings import Settings
import torch
from typing import List
from ultralytics.nn.tasks import PoseModel


def fm_yolo(model: PoseModel, videos: torch.Tensor, settings: Settings) -> List[torch.Tensor]:
    """
    Function to get feature maps from yolo

    Args:
        model: the YOLO model
        videos: the input videos: (bs, seq, c, h, w)
        settings: the settings

    Returns:
        torch.Tensor: feature maps: (bs, seq, chw * layers)
    """
    hooked_modules = settings.yolo_hooks
    n = len(hooked_modules)
    frame_wise = list()
    feature_maps = [torch.Tensor([]).to(settings.train_dev) for _ in range(n)]
    C3k2_hooks = [model.model.model[mi].register_forward_hook(
        lambda module, input, output : frame_wise.append(output)
    ) for mi in hooked_modules]

    for frame in range(videos.shape[1]):
        model.predict(videos[:,frame,:,:,:], verbose=False, imgsz=settings.image_size, device=settings.train_dev, half=True)

        # yolo uses warmup dummies to load everything before inference
        if len(frame_wise) == 2*n:
            frame_wise = frame_wise[n:]

        for i in range(n):
            frame_wise[i] = frame_wise[i].unsqueeze(1) # (bs, c, h, w) -> (bs, seq, c, h, w): inplace crashes: https://github.com/pytorch/rfcs/pull/17
            feature_maps[i] = torch.cat((feature_maps[i], frame_wise[i]), dim=1)
        frame_wise.clear()

    for C3k2_hook in C3k2_hooks:
        C3k2_hook.remove()
    return feature_maps


if __name__ == "__main__":
    raise NotImplementedError("Usage: python main.py args")