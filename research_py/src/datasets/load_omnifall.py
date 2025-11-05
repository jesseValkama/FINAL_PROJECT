from datasets import load_dataset
import numpy as np
import pandas as pd
from src.settings import Settings
from typing import Dict


def load_omnifall_info(settings: Settings) -> Dict:
    """
    loads the omnifall dataset

    splits:
        - cs: regular cs split (subject wise)
        - cv: regular cv split (camera wise)
        - cv-staged -> Only lab datasets
        - cs-staged -> Only lab datasets
        - cv-staged-wild -> Lab datasets for train and val, only OOPS-Fall in test set
        - cs-staged-wild -> Lab datasets for train and val, only OOPS-Fall in test set
    
    Acknowledgements:
        https://arxiv.org/pdf/2505.19889
        https://huggingface.co/datasets/simplexsigil2/omnifall

    """
    valid = ["cs", "cv", "cs-staged", "cv-staged", "cs-staged-wild", "cv-staged-wild"]
    assert settings.split_format in valid, "Please enter a valid split format"
    labels = load_dataset("simplexsigil2/omnifall", "labels")["train"] # train contains all labels
    data = load_dataset("simplexsigil2/omnifall", settings.split_format)
    labels_df = pd.DataFrame(labels)
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

    for _, split_data in [("CS", data)]:
        for subset_name, subset in split_data.items():
            assert subset_name in ds_info, "incorrect keys"
            subset_df = pd.DataFrame(subset)
            merged_df = pd.merge(subset_df, labels_df, on="path", how="left")
            merged_df = merged_df[(merged_df["dataset"].isin(settings.omnifall_subsets))
                                   & (merged_df["label"] < 10) # labels over 10 are only for oops -> staged-to-wild wouldnt work
                                   & (~merged_df["path"].isin(settings.omnifall_corrupt_clips))]
            if subset_name == "test":
                merged_df = merged_df[:1460] 
            print(f"{subset_name} split: {len(merged_df)} clips with labels")
            set_samples = np.array([])
            for i in range(len(settings.dataset_labels)):
                n = len(merged_df[merged_df["label"] == i])
                set_samples = np.append(set_samples, n)
                print(f"  {settings.dataset_labels[i]}: {n}")

            set_paths = merged_df["path"].to_list()
            set_datasets = merged_df["dataset"].to_list()
            set_times = merged_df[["start", "end"]].to_numpy()
            set_labels = merged_df["label"].to_numpy(dtype=np.uint8)
            assert len(set_paths) == len(set_labels), "the data is corrupt"
            ds_info[subset_name]["paths"] = set_paths
            ds_info[subset_name]["datasets"] = set_datasets
            ds_info[subset_name]["times"] = set_times
            ds_info[subset_name]["labels"] = set_labels
            ds_info[subset_name]["samples"] = set_samples
    return ds_info


if __name__ == "__main__":
    raise NotImplementedError("Usage: python main.py args")