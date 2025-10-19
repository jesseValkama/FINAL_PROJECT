import csv
import numpy as np
import os
from pathlib import Path
from src.settings.settings import Settings
from typing import Dict


def load_ucf101_info(settings: Settings) -> Dict:
    """
    Loads the UCF50 dataset

    Args:    

    Returns:

    """

    ds_info = {
        "train": {
            "paths": None,
            "datasets": None,
            "times": None,
            "labels": None,
            "samples": None
        },
        "validation": {
            "paths": None,
            "datasets": None,
            "times": None,
            "labels": None,
            "samples": None
        },
        "test": {
            "paths": None,
            "datasets": None,
            "times": None,
            "labels": None,
            "samples": None
        }
    }
    ds = "UCF101" # weird hard coding but works for now, to normalise formats with omnifall
    path = Path(settings.dataset_path + "/" + ds)
    for split in ["train", "val", "test"]:
        file_path = path / (split + ".csv")
        assert file_path.exists()

        set_paths = list()
        set_datasets = list()
        set_times = np.array([])
        set_labels = np.array([])
        set_samples = np.array([])

        with open(file_path) as file:
            reader = csv.DictReader(file)
            for row in reader:
                set_paths.append(row["clip_path"])
                set_datasets.append(ds)
                set_times = ...
                set_lables = np.append(set_labels, row["label"])
                set_samples = ...

    return ds_info


        