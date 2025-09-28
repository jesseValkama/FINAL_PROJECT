from src.datasets import get_omnifall_datasets, get_data_loader
from torch.utils.data import DataLoader
from src.train.predict_keypoints import kp_yolo


def run_loop(settings) -> None:
    """
    main training loop fn that calls train and test

    Args:

    Returns:

    """
    train_set, val_set, test_set = get_omnifall_datasets(settings)

    if settings.train:
        train_loader = get_data_loader(train_set, settings.train_batch_size, True, settings.num_workers)
        val_loader = get_data_loader(val_set, settings.val_batch_size, True, settings.num_workers)
        train(train_loader, val_loader, settings)

    if settings.test:
        test_loader = get_data_loader(test_set, settings.test_batch_size, True, settings.num_workers)
        test(test_loader, settings)


def train(train_loader, val_loader, settings) -> None:
    """
    Training fn that calls validate

    Args:

    Returns:

    """
    
    for data in train_loader:
        vids, labels = data
        # vids, labels = vids.to(dev), labels.to(dev)
        kps = kp_yolo(settings, vids)


def validate(test_loader, settings) -> None:
    """
    Validation fn

    Args:

    Returns:

    """
    ...


def test() -> None:
    """
    Testing fn

    Args:

    Returns:

    """
    ...


if __name__ == "__main__":
    raise NotImplementedError("Usage: python main.py args")