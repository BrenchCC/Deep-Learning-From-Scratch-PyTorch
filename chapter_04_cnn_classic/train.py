import os
import sys
import logging
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.append(os.getcwd())

from chapter_04_cnn_classic.model import SimpleCNN

from utils import get_device
from utils import save_json
from utils import setup_seed
from utils import CIFAR10_STATS
from utils import log_model_info
from utils import run_classification_epoch

logger = logging.getLogger("TrainCIFAR")


def parse_args():
    """
    Parse command-line arguments for chapter 04 training.

    Args:
        None

    Returns:
        argparse.Namespace: Parsed training configuration.
    """
    parser = argparse.ArgumentParser(description = "Train SimpleCNN on CIFAR-10")
    parser.add_argument("--epochs", type = int, default = 10, help = "Number of training epochs")
    parser.add_argument("--batch_size", type = int, default = 128, help = "Batch size")
    parser.add_argument("--lr", type = float, default = 0.001, help = "Learning rate")
    parser.add_argument("--seed", type = int, default = 42, help = "Random seed")
    parser.add_argument("--num_workers", type = int, default = 4, help = "DataLoader workers")
    parser.add_argument("--data_dir", type = str, default = "./chapter_04_cnn_classic/data", help = "Data directory")
    parser.add_argument("--save_dir", type = str, default = "./chapter_04_cnn_classic/results", help = "Save directory")
    return parser.parse_args()


def args_parser():
    """
    Backward-compatible alias for parse_args.

    Args:
        None

    Returns:
        argparse.Namespace: Parsed training configuration.
    """
    return parse_args()


def attach_file_handler(log_path):
    """
    Attach a file log handler once to the root logger.

    Args:
        log_path (str): Target log file path.

    Returns:
        None
    """
    abs_log_path = os.path.abspath(log_path)
    root_logger = logging.getLogger()

    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == abs_log_path:
            return

    file_handler = logging.FileHandler(abs_log_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    root_logger.addHandler(file_handler)


def build_dataloaders(data_root, batch_size, num_workers):
    """
    Build CIFAR-10 train and test DataLoaders.

    Args:
        data_root (str): Dataset root path.
        batch_size (int): Batch size.
        num_workers (int): Number of DataLoader workers.

    Returns:
        tuple: (train_loader, test_loader, classes)
    """
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*CIFAR10_STATS)
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*CIFAR10_STATS)
    ])

    train_set = datasets.CIFAR10(
        root = data_root,
        train = True,
        download = True,
        transform = train_transform
    )
    test_set = datasets.CIFAR10(
        root = data_root,
        train = False,
        download = True,
        transform = test_transform
    )

    use_persistent_workers = num_workers > 0
    train_loader = DataLoader(
        train_set,
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers,
        pin_memory = True,
        persistent_workers = use_persistent_workers
    )
    test_loader = DataLoader(
        test_set,
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers,
        pin_memory = True,
        persistent_workers = use_persistent_workers
    )
    return train_loader, test_loader, train_set.classes


def build_training_components(lr, epochs, device):
    """
    Build model and optimization components.

    Args:
        lr (float): Learning rate.
        epochs (int): Number of epochs for scheduler settings.
        device (torch.device): Runtime device.

    Returns:
        tuple: (model, criterion, optimizer, scheduler)
    """
    model = SimpleCNN(num_classes = 10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr = lr, weight_decay = 1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs, eta_min = 1e-6)
    return model, criterion, optimizer, scheduler


def main():
    """
    Main training entry for chapter 04.

    Args:
        None

    Returns:
        None
    """
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok = True)
    attach_file_handler(os.path.join(args.save_dir, "train_log.txt"))

    setup_seed(args.seed)
    device = get_device()
    logger.info(f"Using device: {device}")
    logger.info(f"Hyperparameters: {vars(args)}")

    train_loader, test_loader, classes = build_dataloaders(
        data_root = args.data_dir,
        batch_size = args.batch_size,
        num_workers = args.num_workers
    )
    logger.info(f"Classes: {classes}")

    model, criterion, optimizer, scheduler = build_training_components(
        lr = args.lr,
        epochs = args.epochs,
        device = device
    )
    log_model_info(model)

    metrics = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "learning_rate": []
    }
    best_acc = 0.0

    logger.info("Starting training...")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_classification_epoch(
            model = model,
            dataloader = train_loader,
            criterion = criterion,
            device = device,
            stage = "train",
            optimizer = optimizer,
            epoch_idx = epoch
        )
        val_metrics = run_classification_epoch(
            model = model,
            dataloader = test_loader,
            criterion = criterion,
            device = device,
            stage = "eval",
            epoch_idx = epoch
        )

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        metrics["train_loss"].append(train_metrics["loss"])
        metrics["train_acc"].append(train_metrics["acc"])
        metrics["val_loss"].append(val_metrics["loss"])
        metrics["val_acc"].append(val_metrics["acc"])
        metrics["learning_rate"].append(current_lr)

        logger.info(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['acc']:.2f}% | "
            f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['acc']:.2f}% | "
            f"LR: {current_lr:.6f}"
        )

        if val_metrics["acc"] > best_acc:
            best_acc = val_metrics["acc"]
            save_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            logger.info(f"New best model saved with Acc: {best_acc:.2f}%")

    metrics_path = os.path.join(args.save_dir, "metrics.json")
    save_json(metrics, metrics_path)
    logger.info(f"Training completed. Best Acc: {best_acc:.2f}%")
    logger.info(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers = [logging.StreamHandler()]
    )
    main()
