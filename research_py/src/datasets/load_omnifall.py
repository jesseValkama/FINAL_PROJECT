from datasets import load_dataset
import pandas as pd
from typing import Dict 


def load_omnifall_info(settings) -> Dict:
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
            "paths": list(),
            "datasets": list(),
            "times": list(),
            "labels": list()
        },
        "validation": {
            "paths": list(),
            "datasets": list(),
            "times": list(),
            "labels": list()
        },
        "test": {
            "paths": list(),
            "datasets": list(),
            "times": list(),
            "labels": list()
        }
    }    

    for _, split_data in [("CS", data)]:
        for subset_name, subset in split_data.items():
            assert subset_name in ds_info, "incorrect keys"
            subset_df = pd.DataFrame(subset)

            merged_df = pd.merge(subset_df, labels_df, on="path", how="left")
            merged_df = merged_df[merged_df["dataset"] == "le2i"]
            print(f"  {subset_name} split: {len(merged_df)} clips with labels")

            set_paths = merged_df["path"].to_list()
            set_datasets = merged_df["dataset"].to_list()
            set_times = merged_df[["start", "end"]].to_numpy()
            
            set_labels = merged_df["label"].to_numpy()
            assert len(set_paths) == len(set_labels), "the data is corrupt"

            ds_info[subset_name]["paths"].extend(set_paths)
            ds_info[subset_name]["datasets"].extend(set_datasets)
            ds_info[subset_name]["times"].extend(set_times)
            ds_info[subset_name]["labels"].extend(set_labels)

    return ds_info


if __name__ == "__main__":
    raise NotImplementedError("Usage: python main.py args")