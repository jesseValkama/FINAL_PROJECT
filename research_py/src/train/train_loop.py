from datetime import datetime as datetime # who created this language
import math
import numpy as np
import os
from src.datasets.get_data_loader import get_data_loader
from src.datasets.omnifall import get_omnifall_datasets
from src.models.lstm import LSTM
from src.settings import Settings
from src.train.early_stop import EarlyStop
from src.train.predict_keypoints import kp_yolo
from src.train.process_keypoints import kp_process
import time
import torch
from torch.amp import GradScaler, autocast
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from ultralytics import YOLO


def run_loop(settings: Settings) -> None:
    """
    main training loop fn that calls train and test

    Args:

    Returns:

    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter("train".format(timestamp))

    train_set, val_set, test_set = get_omnifall_datasets(settings)

    if settings.train:
        train_loader = get_data_loader(train_set, settings.train_batch_size, True, settings.num_workers)
        val_loader = get_data_loader(val_set, settings.val_batch_size, True, settings.num_workers)
        train(train_loader, val_loader, writer, settings)

    if settings.test:
        test_loader = get_data_loader(test_set, settings.test_batch_size, True, settings.num_workers)
        test(test_loader, writer, settings)


def train(train_loader: DataLoader, val_loader: DataLoader, writer: SummaryWriter, settings: Settings) -> None:
    """
    Training fn that calls validate

    Args:

    Returns:

    """
    yolo_path = os.path.join(settings.weights_path, "ultralytics", settings.yolo_model)
    yolo = YOLO(yolo_path) # yolo is automatically moved to the correct dev

    lstm = LSTM(settings=settings)
    lstm = lstm.to(settings.train_dev)

    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW(lstm.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    cosine_annealing = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, settings.max_epochs)

    training_loss = 0.0
    val_loss = 0.0
    best_val_loss = np.inf
    early_stop = EarlyStop(settings.min_epochs, settings.early_stop_tries)

    start_time = time.time()
    for epoch in range(settings.max_epochs):

        print(f"Starting epoch: {epoch}")
        training_loss = 0.0

        for i, (vids, labels) in enumerate(train_loader):
            
            # vids are automatically moved to the correct dev by the cnn model
            labels = labels.to(settings.train_dev)
            kps = kp_yolo(yolo, vids, settings)

            kp_process()

            optimiser.zero_grad()

            outputs = lstm(kps)
            loss = criterion(outputs, labels.view(-1))
            loss.backward()
            optimiser.step()
            iter_loss = loss.item()
            assert not math.isnan(iter_loss), "Training is unstable, change settings"
            training_loss += iter_loss

        training_loss /= i
        print("The training loss is: ", training_loss)
        print("The training accuracy is: ")

        if epoch % settings.validation_interval == 0 and epoch != 0:
            val_loss = validate(lstm, yolo, val_loader, writer, criterion, settings)
            improvement = False
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                improvement = True
                print("The model improved")
                torch.save(lstm, settings.work_model + ".pth")
            if early_stop(epoch, improvement):
                break
                
        cosine_annealing.step()

    end_time = time.time()
    training_time = end_time - start_time
    print("The training took: ", training_time)


@torch.no_grad
def validate(lstm: LSTM, yolo, val_loader: DataLoader, writer: SummaryWriter, criterion: nn.CrossEntropyLoss, settings: Settings) -> float:
    """
    Validation fn

    Args:

    Returns:

    """
    validation_loss = 0.0
    for i, (videos, labels) in enumerate(val_loader):
        labels = labels.to(settings.train_dev)
        kps = kp_yolo(yolo, videos, settings)

        kp_process()

        outputs = lstm(kps)
        loss = criterion(outputs, labels.view(-1))
        iter_loss = loss.item()
        assert not math.isnan(iter_loss), "validation is unstable, change settings"
        validation_loss += iter_loss

    validation_loss /= i
    print("The validation loss is: ", validation_loss)
    print("The validation accuracy is: ")

    return validation_loss


@torch.no_grad
def test() -> None:
    """
    Testing fn

    Args:

    Returns:

    """
    ...


if __name__ == "__main__":
    raise NotImplementedError("Usage: python main.py args")