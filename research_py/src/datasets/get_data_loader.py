from torch.utils.data import DataLoader


def get_data_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, 
                    pin_memory: bool=False, persistent_workers: bool=True, 
                    multiprocessing_context: str="spawn") -> DataLoader:
    """
    Function for creating a dataloader, pin_memory is false by default, since
    the images DO NOT need to be moved to vram due to pretraining

    Args:
        dataset: the dataset for the loader
        batch_size: the amount of data per forward pass
        shuffle: random batches
        num_workers: the n of cores for multiprocessing
        pin_memory: basically enables async transfers to gpu if non_blocking = True
        persistent_workers: whether to keep the workers after each epoch
        multiprocessing_context: specifies the way to start multiprocessing
    """
    persistent_workers = persistent_workers if num_workers > 0 else False
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )


if __name__ == "__main__":
    raise NotImplementedError("Usage: python main.py args")