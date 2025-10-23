import math
import numpy as np
import os
from src.datasets.get_data_loader import get_data_loader
from src.datasets.load_omnifall import load_omnifall_info
from src.datasets.load_ucf101_info import load_ucf101_info
from src.datasets.omnifall import get_omnifall_datasets
from src.metrics.metrics_container import MetricsContainer
from src.models.efficientnet_lrcn import EfficientLRCN
from src.settings.settings import Settings
from src.utils.balance import get_cls_weights 
from src.utils.early_stop import EarlyStop
from src.visualise.plot_container import PlotContainer
import time
import torch
from torch.amp import autocast, GradScaler
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def run_loop(settings: Settings) -> None:
    """
    main training loop fn that calls train and test

    Args:

    Returns:

    """
    writer = SummaryWriter()
    plot_container = PlotContainer(writer, settings)
    info_fn = load_omnifall_info if settings.dataset == "omnifall" else load_ucf101_info
    ds_info = info_fn(settings)
    samples = ds_info["train"]["samples"]
    train_set, val_set, test_set = get_omnifall_datasets(ds_info, settings)
    if settings.train:
        train_loader = get_data_loader(train_set, settings.train_batch_size, True, settings.num_workers, settings.async_transfers)
        val_loader = get_data_loader(val_set, settings.val_batch_size, True, settings.num_workers, settings.async_transfers)
        train(train_loader, val_loader, samples, plot_container, settings)
    if settings.test:
        test_loader = get_data_loader(test_set, settings.test_batch_size, True, settings.num_workers, settings.async_transfers)
        test(test_loader, plot_container, settings)
       

def train(train_loader: DataLoader, val_loader: DataLoader, samples: np.ndarray, plot_container: PlotContainer, settings: Settings) -> None:
    """
    Training fn that calls validate

    Args:

    Returns:

    """
    model = EfficientLRCN(settings)
    model = model.to(settings.train_dev)
    cls_weights = get_cls_weights(samples, settings)
    criterion = nn.CrossEntropyLoss(weight=cls_weights, label_smoothing=settings.label_smoothing)
    optimiser = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    cosine_annealing = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, settings.max_epochs)
    # TODO: implement warmup
    scaler = GradScaler()
    training_loss = 0.0
    val_loss = 0.0
    best_val_loss = np.inf
    early_stop = EarlyStop(settings.min_epochs, settings.early_stop_tries)
    start_time = time.time()

    for epoch in range(1, settings.max_epochs+1):
        print(f"Starting epoch: {epoch}")
        training_loss = 0.0
        train_correct = 0.0
        train_total = 0.0
        model.train()

        for i, (vids, labels) in enumerate(train_loader):
            vids, labels = vids.to(settings.train_dev, non_blocking=settings.async_transfers), labels.to(settings.train_dev, non_blocking=settings.async_transfers).view(-1)
            optimiser.zero_grad()
            with autocast(device_type="cuda"):
                outputs = model(vids)
                loss = criterion(outputs, labels)
            idxs = torch.argmax(outputs, dim=1)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            iter_loss = loss.item()
            assert not math.isnan(iter_loss), "Training is unstable, change settings"
            training_loss += iter_loss
            for idx in range(len(idxs)):
                train_total += 1.0
                if idxs[idx] == labels[idx]:
                    train_correct += 1.0

        training_loss /= (i + 1)
        training_accuracy = train_correct / train_total
        print("The training loss is: ", training_loss)
        print("The training accuracy is: ", training_accuracy)
        plot_container.update_train_plots(training_loss, training_accuracy, "train")
        if epoch % settings.validation_interval == 0 and epoch != 0:
            val_loss = validate(model, val_loader, plot_container, criterion, settings)
            improvement = False
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                improvement = True
                print("The model improved")
                save_path = os.path.join(settings.weights_path, settings.work_model + ".pth")
                torch.save(model, save_path)
            if early_stop(epoch, improvement):
                break
        cosine_annealing.step()

    plot_container.push_train_plots()
    end_time = time.time()
    training_time = end_time - start_time
    print("The training took: ", training_time)


@torch.no_grad
def validate(model: EfficientLRCN, val_loader: DataLoader, plot_container: PlotContainer, criterion: nn.CrossEntropyLoss, settings: Settings) -> float:
    """
    Validation fn

    Args:

    Returns:

    """
    model.eval()
    validation_loss = 0.0
    val_total = 0.0
    val_correct = 0.0

    for i, (vids, labels) in enumerate(val_loader):
        vids, labels = vids.to(settings.train_dev, non_blocking=settings.async_transfers), labels.to(settings.train_dev, non_blocking=settings.async_transfers).view(-1)
        with autocast(device_type="cuda"):
            outputs = model(vids)
            loss = criterion(outputs, labels)
        idxs = torch.argmax(outputs, dim=1)
        iter_loss = loss.item()
        assert not math.isnan(iter_loss), "validation is unstable, change settings"
        validation_loss += iter_loss
        for idx in range(len(idxs)):
            val_total += 1.0
            if idxs[idx] == labels[idx]:
                val_correct += 1.0

    validation_loss /= (i + 1)
    validation_accuracy = val_correct / val_total
    print("The validation loss is: ", validation_loss)
    print("The validation accuracy is: ", validation_accuracy)
    plot_container.update_train_plots(validation_loss, validation_accuracy, "val")
    return validation_loss


@torch.no_grad
def test(test_loader: DataLoader, plot_container: PlotContainer, settings: Settings) -> None:
    """
    Testing fn

    Args:

    Returns:

    """
    print("Starting testing")
    model_name = settings.work_model if settings.train else settings.test_model
    save_path = os.path.join(settings.weights_path, model_name + ".pth")
    model = torch.load(save_path, weights_only=False)
    model.to(settings.train_dev)
    model.eval()
    metrics_container = MetricsContainer(settings.dataset_labels, plot_container)

    for (vids, labels) in test_loader:
        vids, labels = vids.to(settings.train_dev, non_blocking=settings.async_transfers), labels.to(settings.train_dev, non_blocking=settings.async_transfers).view(-1)
        with torch.autocast(device_type="cuda"):
            outputs = model(vids)
        metrics_container.calc_iter(outputs, labels)

    print("\n\n")
    metrics_container.show_conf_mat()
    metrics_container.print_conf_mat() 
    metrics_container.calc_metrics()
    metrics_container.print_metrics()


if __name__ == "__main__":
    raise NotImplementedError("Usage: python main.py args")