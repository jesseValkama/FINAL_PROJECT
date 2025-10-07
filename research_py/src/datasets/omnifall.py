import av
import numpy as np
import os
from pathlib import Path
from src.settings import Settings
import torch
from torchvision import transforms
from typing import Tuple, Dict


def get_omnifall_datasets(ds_info: Dict, settings: Settings) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Function for settings up Omnifall datasets
    Setup transforms in this function

    Args:

    Returns:

    """
    pre_transforms = transforms.Compose([
        transforms.Resize((settings.image_size, settings.image_size))
        # transforms.Normalize(settings.mean, settings.standard_deviation, inplace=True)
    ])
    aug_transforms = transforms.Compose([])

    train = Omnifall(ds_info["train"], settings, pre_transforms, aug_transforms)
    val = Omnifall(ds_info["validation"], settings, pre_transforms)
    test = Omnifall(ds_info["test"], settings, pre_transforms)
    return train, val, test


class Omnifall(torch.utils.data.Dataset):
    """
    Class for handling Omnifall, used by a torch Dataloader
    """
    
    def __init__(self, ds_info: dict, settings: Settings, pre_transforms: transforms.Compose, aug_transforms: transforms.Compose = None) -> None:
        self._video_paths = ds_info["paths"]
        self._video_datasets =  ds_info["datasets"]
        self._video_times = ds_info["times"] 
        self._video_labels = ds_info["labels"]
        self._settings = settings

        self._video_len = settings.video_length
        
        self._pre_transforms = pre_transforms
        self._aug_transforms = aug_transforms

        assert len(self._video_paths) == len(self._video_labels)

    def __len__(self) -> int:
        return len(self._video_labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ext = self._get_ext(idx)
        video_path_str = os.path.join(self._settings.dataset_path, self._video_datasets[idx], self._video_paths[idx] + ext)
        video_path = Path(video_path_str)
        assert video_path.is_file(), f"path to video is invalid {video_path_str}"

        clip = torch.Tensor(self._load_video(video_path))
        clip = clip.permute([0, 3, 1, 2]).contiguous().to(torch.float)
        clip.div_(255.0)
        clip = self._pre_transforms(clip)

        label = self._video_labels[idx]
        label = torch.Tensor([label]).to(torch.long)

        return clip, label
    
    def _get_ext(self, idx) -> str:
        dataset = self._video_datasets[idx]
        match dataset:
            case "le2i":
                return ".avi"
            case _:
                raise RuntimeError(f"Dataset not implemented yet ({dataset})")
            
    def _load_video(self, video_path: Path) -> np.ndarray:
        """
        Function for loading videos, since the default torch codec doesn't work
        due to le2i videos having different fps and sizes and omnifall dataset
        annotation lengths being different lengths -> this custom approach
        """
        fps = 0.0
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            stream.codec_context.skip_frame = "NONKEY"

            fps = float(stream.average_rate)
                        
            capture_interval = round(fps / self._settings.fps)
            video = list() 
            
            for idx, frame in enumerate(container.decode(stream)):
                if idx % capture_interval == 0:
                    img = frame.to_ndarray(format="rgb24")
                    video.append(img) 

        # TODO: SLOW, FIX
        video = np.array(video)
        if len(video) == self._video_len:
            return video
        
        # TODO: temp fix for vids too short 
        if len(video) >= self._video_len:
            indices = np.linspace(0, len(video) - 1, self._video_len, dtype=int)
            return video[indices]
        else:
            pad = [video[-1]] * (self._video_len - len(video))
            return np.concatenate([video, pad], axis=0)


if __name__ == "__main__":
    raise NotImplementedError("Usage: python main.py args")
