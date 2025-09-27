from torch.utils.data import DataLoader


def get_data_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, 
                    pin_memory: bool=True, persistent_workers: bool=True, 
                    multiprocessing_context: str="spawn") -> DataLoader:
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