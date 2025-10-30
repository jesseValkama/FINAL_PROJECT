import albumentations as A
import av
import cv2 as cv
import numpy as np
import os
from pathlib import Path
from src.settings import Settings
import torch
from torchvision.transforms import v2
from typing import Tuple, Dict


def get_omnifall_datasets(ds_info: Dict, settings: Settings) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Function for settings up Omnifall datasets
    Setup transforms in this function

    Args:

    Returns:

    """
    pre_transforms = A.ReplayCompose([
        A.Resize(width=settings.image_size, height=settings.image_size)
    ])
    aug_transforms = A.ReplayCompose([
        A.HorizontalFlip(),
        A.ISONoise(p=0.3),
        A.RandomGamma(p=0.3)
    ])
    post_transforms = v2.Compose([
        v2.Normalize(mean=settings.mean, std=settings.standard_deviation, inplace=True)  
    ])
    train = Omnifall(ds_info["train"], settings, pre_transforms, post_transforms, aug_transforms=aug_transforms)
    val = Omnifall(ds_info["validation"], settings, pre_transforms, post_transforms)
    test = Omnifall(ds_info["test"], settings, pre_transforms, post_transforms)
    return train, val, test


class Omnifall(torch.utils.data.Dataset):
    """
    Class for handling Omnifall, used by a torch Dataloader
    """
    
    def __init__(self, ds_info: dict, settings: Settings, pre_transforms: A.ReplayCompose, post_transforms: v2.Compose, aug_transforms: A.ReplayCompose | None = None) -> None:
        """
        """
        assert isinstance(pre_transforms, A.ReplayCompose) and isinstance(aug_transforms, (A.ReplayCompose, type(None))) and isinstance(post_transforms, v2.Compose)
        self._video_paths = ds_info["paths"]
        self._video_datasets =  ds_info["datasets"]
        self._video_times = ds_info["times"] 
        self._video_labels = ds_info["labels"]
        self._settings = settings
        self._video_len = settings.video_length
        self._pre_transforms = pre_transforms
        self._post_transforms = post_transforms
        self._aug_transforms = aug_transforms
        assert len(self._video_paths) == len(self._video_labels)

    def __len__(self) -> int:
        """
        """
        return len(self._video_labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        """
        ext = self._get_ext(idx)
        video_path_str = os.path.join(self._settings.dataset_path, self._video_datasets[idx], self._video_paths[idx] + ext)
        video_path = Path(video_path_str)
        assert video_path.is_file(), f"path to video is invalid {video_path_str}"

        clip = self._load_video(video_path, self._video_times[idx])
        #clip = self._apply_transforms(clip, self._pre_transforms)
        if self._aug_transforms:
            clip = self._apply_transforms(clip, self._aug_transforms)

        clip = torch.Tensor(clip) 
        clip = clip.permute([0, 3, 1, 2]).contiguous().to(torch.float)
        clip.div_(255.0)
        clip = self._post_transforms(clip)
        label = self._video_labels[idx]
        label = torch.Tensor([label]).to(torch.long)
        return clip, label
    
    def _apply_transforms(self, clip: np.ndarray, transforms: A.Compose) -> torch.Tensor:
        """
        Method for applying pre and aug transforms since Albumentations don't work
        naively with videos

        Args:

        Returns:

        """
        n = len(clip)
        t = transforms(image=clip[0])
        replay = t["replay"]
        transformed = [t["image"]]
        for i in range(1, n):
            transformed.append(A.ReplayCompose.replay(replay, image=clip[i])["image"])
        return np.array(transformed)

    def _get_ext(self, idx: int) -> str:
        """
        Method for getting the ext for a video clip

        Args:

        Returns:

        """
        dataset = self._video_datasets[idx]
        match dataset:
            case "le2i":
                return ".avi"
            case _:
                raise RuntimeError(f"Dataset not implemented yet ({dataset})")
            
    def _load_video(self, video_path: Path, time_steps: np.ndarray) -> np.ndarray:
        """
        Function for loading videos, since the default torch codec doesn't work
        due to le2i videos having different fps, sizes, and omnifall dataset
        annotation lengths being different lengths -> this custom approach

        Args:

        Returns:

        """
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            stream.codec_context.skip_frame = "NONKEY"

            pts = float(stream.time_base)
            start_pts = int(time_steps[0] / pts) # pyav crashes if using np int
            end_pts = int(time_steps[1] / pts)
            container.seek(start_pts, stream=stream)

            assert time_steps[1] > time_steps[0]
            clip_length = (time_steps[1] - time_steps[0]) * float(stream.average_rate)

            # this ends up overshooting, but should be fine for now
            capture_interval = round(clip_length / self._video_len)
            video = list() 
            for i, frame in enumerate(container.decode(stream)):
                if i % capture_interval == 0:
                    img = frame.to_ndarray(format="rgb24")
                    h, w = img.shape[:-1]
                    pad_y = max(0, (w - h) // 2)
                    pad_x = max(0, (h - w) // 2)
                    img = cv.copyMakeBorder(img, pad_y,  pad_y, pad_x, pad_x, cv.BORDER_CONSTANT)
                    img = cv.resize(img, (self._settings.image_size, self._settings.image_size))
                    video.append(img) 
                if frame.pts * stream.time_base > end_pts:
                    break

        video = np.array(video)
        n = len(video)
        if n > self._video_len:
            indices = np.linspace(0, len(video) - 1, self._video_len, dtype=int)
            return video[indices]
        elif n < self._video_len:
            pad = [video[-1]] * (self._video_len - len(video))
            return np.concatenate([video, pad], axis=0)
        return video


if __name__ == "__main__":
    raise NotImplementedError("Usage: python main.py args")
