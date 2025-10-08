from src.settings.settings import Settings
import torch
from ultralytics.nn.tasks import PoseModel


def fm_yolo(model: PoseModel, videos: torch.Tensor, settings: Settings) -> torch.Tensor:
    """
    Function to get feature maps from yolo

    Args:
        model: the YOLO model
        videos: the input videos: (bs, seq, c, h, w)
        settings: the settings

    Returns:
        torch.Tensor: feature maps: (bs, seq, chw * layers)

    """
    head_attached_C3k2 = [16, 19, 22]
    flat_frames = list()
    C3k2_hooks = [model.model.model[l].register_forward_hook(
        lambda module, input, output : flat_frames.append(output.view(output.shape[0], -1))
    ) for l in head_attached_C3k2]

    feature_maps = torch.Tensor([]).to(settings.train_dev)
    for frame in range(videos.shape[1]):
        model.predict(videos[:,frame,:,:,:], verbose=False, imgsz=settings.image_size, device=settings.train_dev, half=True)

        # yolo uses warmup dummies to optimise inference
        if len(flat_frames) == 6:
            flat_frames = flat_frames[3:]

        frame_wise = torch.Tensor([]).to(settings.train_dev)
        for flat_frame in flat_frames:
            frame_wise = torch.cat((frame_wise, flat_frame), dim=1)
        flat_frames.clear()

        frame_wise.unsqueeze_(1) # (bs, chw) -> (bs, seq, chw)
        feature_maps = torch.cat((feature_maps, frame_wise), dim=1)

    for C3k2_hook in C3k2_hooks:
        C3k2_hook.remove()
    return feature_maps


if __name__ == "__main__":
    raise NotImplementedError("Usage: python main.py args")