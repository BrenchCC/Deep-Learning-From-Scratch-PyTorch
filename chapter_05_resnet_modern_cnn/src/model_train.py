"""
model_train.py
Conducts the A/B test: ResNet-18 vs. PlainNet-18 on STL-10.
Tracks training dynamics and visualizes the 'Degradation Problem' (or lack thereof due to depth).
"""

import os
import sys
import json
import logging
import argparse
from unittest import loader

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.append(os.getcwd())

from chapter_04_cnn_classic import model
from chapter_05_resnet_modern_cnn.src.model import resnet18, resnet34
from chapter_05_resnet_modern_cnn.src.dataset import get_stl10_loaders

try:
    from utils import get_device, setup_seed, Timer, save_json, count_parameters
except ImportError:
    # Fallback if utils not present
    def get_device(): 
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    def setup_seed(seed): torch.manual_seed(seed); np.random.seed(seed)
    class Timer: 
        def __enter__(self): pass 
        def __exit__(self, *args): pass
    def log_model_info(model): pass

logger = logging.getLogger("Model Training")

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device 
):
    """
    Train the model for one epoch.

    Args:
        model: The neural network.
        loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimization algorithm.
        device: Device to run on (CPU/CUDA/MPS).
    
    Returns:
        float: Average loss for this epoch.
        float: Accuracy percentage for this epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc = "Training", leave = False)
    
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({"loss": loss.item()})

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def evaluate(
    model,
    loader,
    criterion,
    device
):
    """
    Evaluate the model on the validation set.

    Args:
        model: The neural network.
        loader: DataLoader for validation data.
        criterion: Loss function.
        device: Device to run on.
    
    Returns:
        float: Average loss.
        float: Accuracy percentage.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def run_experiment(model_name, use_residual, loaders, args, device):
    """
    Execute a full training session for a given model configuration.

    Args:
        model_name (str): Name of the model (for logging/saving).
        use_residual (bool): Whether to use residual connections.
        loaders (tuple): (train_loader, test_loader, classes).
        args (Namespace): Parsed command line arguments.
        device: Device object.
    
    Returns:
        dict: Training history (loss and accuracy curves).
    """
    logger.info(f"--- Starting Experiment: {model_name} (Residual={use_residual}) ---")
    
    # Initialize Model
    model = None
    if model_name.lower() == "resnet18" or model_name.lower() == "plainnet18":
        model = resnet18(num_classes = 10, use_residual = use_residual)
    elif model_name.lower() == "resnet34" or model_name.lower() == "plainnet34":
        model = resnet34(num_classes = 10, use_residual = use_residual)

    model = model.to(device)
    
    logger.info(f"Model Parameters: {count_parameters(model):,}")

    # Optimizer & Scheduler
    optimizer = optim.SGD(
        model.parameters(), 
        lr = args.lr, 
        momentum = 0.9, 
        weight_decay = 5e-4
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.1)
    criterion = nn.CrossEntropyLoss()

    train_loader, test_loader, _ = loaders
    
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": []
    }

    timer = Timer()
    timer.start()

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        logger.info(
            f"Epoch [{epoch+1}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%"
        )
    
    time_cost = timer.stop()
    logger.info(f"Finished {model_name} in {time_cost:.2f}s")

    # Save Checkpoint
    os.makedirs(args.checkpoint_dir, exist_ok = True)
    save_path = os.path.join(args.checkpoint_dir, f"{model_name.lower()}_stl10.pth")
    torch.save(model.state_dict(), save_path)
    logger.info(f"Model saved to {save_path}")

    return history

def plot_comparison(resnet_hist, plain_hist, save_dir):
    """
    Plot and save the comparison between ResNet and PlainNet.
    """
    epochs = range(1, len(resnet_hist["train_loss"]) + 1)
    
    plt.figure(figsize = (12, 5))
    
    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, resnet_hist["train_loss"], 'r-', label = 'ResNet-18 Train')
    plt.plot(epochs, plain_hist["train_loss"], 'b--', label = 'PlainNet-18 Train')
    plt.title('Training Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha = 0.3)

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, resnet_hist["val_acc"], 'r-', label = 'ResNet-18 Val Acc')
    plt.plot(epochs, plain_hist["val_acc"], 'b--', label = 'PlainNet-18 Val Acc')
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha = 0.3)

    os.makedirs(save_dir, exist_ok = True)
    save_path = os.path.join(save_dir, "loss_comparison.png")
    plt.savefig(save_path)
    logger.info(f"Comparison plot saved to {save_path}")

def args_parser():
    parser = argparse.ArgumentParser(description = "Chapter 05: ResNet vs PlainNet on STL-10")
    
    parser.add_argument('--mode', type = str, default = 'compare', choices = ['compare', 'resnet', 'plain'],
                        help = 'Execution mode: compare (both), resnet (only), or plain (only)')
    parser.add_argument('--epochs', type = int, default = 15,
                        help = 'Number of training epochs')
    parser.add_argument('--batch_size', type = int, default = 128,
                        help = 'Batch size for training and evaluation')
    parser.add_argument('--lr', type = float, default = 0.01,
                        help = 'Initial learning rate')
    parser.add_argument('--data_root', type = str, default = 'chapter_05_resnet_modern_cnn/data',
                        help = 'Root directory for dataset')
    parser.add_argument('--result_dir', type = str, default = 'chapter_05_resnet_modern_cnn/results',
                        help = 'Directory to save plots and logs')
    parser.add_argument('--checkpoint_dir', type = str, default = 'chapter_05_resnet_modern_cnn/checkpoints',
                        help = 'Directory to save model weights')
    parser.add_argument('--seed', type = int, default = 42,
                        help = 'Random seed for reproducibility')

    args = parser.parse_args()

    return args

def model_train():
    """
    Main execution entry point with argument parsing.
    """

    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler(sys.stdout)]
    )

    # Parse arguments
    args = args_parser()

    # basic setup
    setup_seed(args.seed)
    device = get_device()
    logger.info(f"Arguments: {args}")
    logger.info(f"Device: {device}")

    # data loading
    loaders = get_stl10_loaders(data_root = args.data_root, batch_size = args.batch_size)

    # Execute experiments based on mode 
    resnet_history = None
    plain_history = None

    if args.mode in ['compare', 'resnet']:
        resnet_history = run_experiment("ResNet18", True, loaders, args, device)
    
    if args.mode in ['compare', 'plain']:
        plain_history = run_experiment("PlainNet18", False, loaders, args, device)

    # 6. Visualization (Only if comparison mode is active)
    if args.mode == 'compare' and resnet_history and plain_history:
        plot_comparison(resnet_history, plain_history, args.result_dir)

if __name__ == "__main__":
    model_train()