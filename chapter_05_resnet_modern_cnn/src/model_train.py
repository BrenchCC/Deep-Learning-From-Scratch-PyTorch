"""
A/B training for ResNet and PlainNet on STL-10.
"""

import os
import sys
import logging
import argparse

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.getcwd())

from chapter_05_resnet_modern_cnn.src.dataset import get_stl10_loaders
from chapter_05_resnet_modern_cnn.src.model import resnet18, resnet34

from utils import get_device
from utils import save_json
from utils import setup_seed
from utils import count_parameters
from utils import run_classification_epoch

logger = logging.getLogger("ModelTraining")


def parse_args():
    """
    Parse command-line arguments for chapter 05 training.

    Args:
        None

    Returns:
        argparse.Namespace: Parsed training configuration.
    """
    parser = argparse.ArgumentParser(description = "Chapter 05: ResNet vs PlainNet on STL-10")
    parser.add_argument("--mode", type = str, default = "compare", choices = ["compare", "resnet", "plain"], help = "Run compare, resnet only, or plain only")
    parser.add_argument("--epochs", type = int, default = 15, help = "Number of training epochs")
    parser.add_argument("--batch_size", type = int, default = 128, help = "Batch size")
    parser.add_argument("--num_workers", type = int, default = 2, help = "DataLoader workers")
    parser.add_argument("--lr", type = float, default = 0.01, help = "Initial learning rate")
    parser.add_argument("--seed", type = int, default = 42, help = "Random seed")
    parser.add_argument("--depth", type = int, default = 18, choices = [18, 34], help = "Model depth")
    parser.add_argument("--data_root", type = str, default = "chapter_05_resnet_modern_cnn/data", help = "Dataset root")
    parser.add_argument("--result_dir", type = str, default = "chapter_05_resnet_modern_cnn/results", help = "Directory to save plots and metrics")
    parser.add_argument("--checkpoint_dir", type = str, default = "chapter_05_resnet_modern_cnn/checkpoints", help = "Directory to save checkpoints")
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


def build_model(depth, use_residual):
    """
    Build ResNet/PlainNet model by depth and residual toggle.

    Args:
        depth (int): Supported values are 18 and 34.
        use_residual (bool): Whether to enable residual shortcuts.

    Returns:
        nn.Module: Built model.
    """
    if depth == 18:
        return resnet18(num_classes = 10, use_residual = use_residual)
    if depth == 34:
        return resnet34(num_classes = 10, use_residual = use_residual)
    raise ValueError(f"Unsupported depth: {depth}")


def run_experiment(model_name, use_residual, loaders, args, device):
    """
    Run one full training experiment and save metrics/checkpoint.

    Args:
        model_name (str): Name prefix used in logs and outputs.
        use_residual (bool): Residual mode flag.
        loaders (tuple): (train_loader, val_loader, classes).
        args (argparse.Namespace): Runtime arguments.
        device (torch.device): Runtime device.

    Returns:
        dict: Training history.
    """
    logger.info(f"Starting experiment: {model_name} | use_residual = {use_residual}")

    model = build_model(depth = args.depth, use_residual = use_residual).to(device)
    logger.info(f"Model Parameters: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum = 0.9, weight_decay = 5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.1)

    train_loader, val_loader, _ = loaders
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "learning_rate": []
    }

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
            dataloader = val_loader,
            criterion = criterion,
            device = device,
            stage = "eval",
            epoch_idx = epoch
        )

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["learning_rate"].append(current_lr)

        logger.info(
            f"{model_name} Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['acc']:.2f}% | "
            f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['acc']:.2f}% | "
            f"LR: {current_lr:.6f}"
        )

    os.makedirs(args.checkpoint_dir, exist_ok = True)
    checkpoint_path = os.path.join(args.checkpoint_dir, f"{model_name.lower()}_stl10.pth")
    torch.save(model.state_dict(), checkpoint_path)
    logger.info(f"Checkpoint saved to {checkpoint_path}")

    metrics_path = os.path.join(args.result_dir, f"{model_name.lower()}_metrics.json")
    save_json(history, metrics_path)
    logger.info(f"Metrics saved to {metrics_path}")
    return history


def plot_comparison(resnet_history, plain_history, save_dir):
    """
    Plot train loss and validation accuracy comparison.

    Args:
        resnet_history (dict): Metrics from residual model.
        plain_history (dict): Metrics from plain model.
        save_dir (str): Output directory.

    Returns:
        str: Saved figure path.
    """
    epochs = range(1, len(resnet_history["train_loss"]) + 1)

    plt.figure(figsize = (12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, resnet_history["train_loss"], "r-", label = "ResNet Train")
    plt.plot(epochs, plain_history["train_loss"], "b--", label = "PlainNet Train")
    plt.title("Training Loss Comparison")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha = 0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, resnet_history["val_acc"], "r-", label = "ResNet Val Acc")
    plt.plot(epochs, plain_history["val_acc"], "b--", label = "PlainNet Val Acc")
    plt.title("Validation Accuracy Comparison")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, alpha = 0.3)

    os.makedirs(save_dir, exist_ok = True)
    save_path = os.path.join(save_dir, "loss_comparison.png")
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Comparison plot saved to {save_path}")
    return save_path


def main():
    """
    Main training entry for chapter 05.

    Args:
        None

    Returns:
        None
    """
    args = parse_args()
    setup_seed(args.seed)
    device = get_device()

    os.makedirs(args.result_dir, exist_ok = True)
    os.makedirs(args.checkpoint_dir, exist_ok = True)

    logger.info(f"Device: {device}")
    logger.info(f"Arguments: {vars(args)}")

    loaders = get_stl10_loaders(
        data_root = args.data_root,
        batch_size = args.batch_size,
        num_workers = args.num_workers
    )

    resnet_history = None
    plain_history = None

    if args.mode in ["compare", "resnet"]:
        model_name = f"resnet{args.depth}"
        resnet_history = run_experiment(
            model_name = model_name,
            use_residual = True,
            loaders = loaders,
            args = args,
            device = device
        )

    if args.mode in ["compare", "plain"]:
        model_name = f"plainnet{args.depth}"
        plain_history = run_experiment(
            model_name = model_name,
            use_residual = False,
            loaders = loaders,
            args = args,
            device = device
        )

    if args.mode == "compare" and resnet_history is not None and plain_history is not None:
        comparison_plot_path = plot_comparison(resnet_history, plain_history, args.result_dir)
        save_json(
            {
                "mode": args.mode,
                "depth": args.depth,
                "comparison_plot": comparison_plot_path,
                "resnet_metrics": resnet_history,
                "plain_metrics": plain_history
            },
            os.path.join(args.result_dir, "compare_metrics.json")
        )


def model_train():
    """
    Backward-compatible alias for main.

    Args:
        None

    Returns:
        None
    """
    main()


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers = [logging.StreamHandler(sys.stdout)]
    )
    main()
