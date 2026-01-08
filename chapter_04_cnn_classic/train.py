import os
import sys
import json
import logging
import argparse
from datetime import datetime

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())

try:
    from utils import get_device, setup_seed, Timer, log_model_info
except ImportError:
    # Fallback if utils not present
    def get_device(): 
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    def setup_seed(seed): torch.manual_seed(seed); np.random.seed(seed)
    class Timer: 
        def __enter__(self): pass 
        def __exit__(self, *args): pass
    def log_model_info(model): pass

from chapter_04_cnn_classic.model import SimpleCNN

# Initialize global logger variable (config in main)
logger = logging.getLogger("TrainCIFAR")

def args_parser():
    parser = argparse.ArgumentParser(description = "Train SimpleCNN on CIFAR-10")
    parser.add_argument("--epochs", type = int, default = 10, help = "Number of training epochs")
    parser.add_argument("--batch_size", type = int, default = 128, help = "Batch size")
    parser.add_argument("--lr", type = float, default = 0.001, help = "Learning rate")
    parser.add_argument("--seed", type = int, default = 42, help = "Random seed")
    parser.add_argument("--data_dir", type = str, default = "./chapter_04_cnn_classic/data", help = "Data directory")
    parser.add_argument("--save_dir", type = str, default = "./chapter_04_cnn_classic/results", help = "Save directory")
    return parser.parse_args()

def get_data_loaders(data_root: str, batch_size: int):
    """
    Prepare CIFAR-10 DataLoaders with augmentation.
    
    Args:
        data_root (str): Path to store data.
        batch_size (int): Batch size.
        
    Returns:
        tuple: (train_loader, test_loader, classes)
    """
    # Standard CIFAR-10 normalization stats
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    # Test/Validation transform
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    # Download and load datasets
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
    
    train_loader = DataLoader(
        train_set, 
        batch_size = batch_size, 
        shuffle = True, 
        num_workers = 2
    )
    test_loader = DataLoader(
        test_set, 
        batch_size = batch_size, 
        shuffle = False, 
        num_workers = 2
    )
    
    return train_loader, test_loader, train_set.classes

# Training function
def train_one_epoch(model, loader, criterion, optimizer, device, epoch_idx):
    """
    Run training for one epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Use tqdm for progress visualization
    loop = tqdm(loader, desc = f"Epoch {epoch_idx} [Train]", leave = False)
    
    for batch_idx, (inputs, targets) in enumerate(loop):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        loop.set_postfix(loss = loss.item(), acc = 100. * correct / total)
        
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# Evaluation function
def evaluate(model, loader, criterion, device, epoch_idx):
    """
    Run evaluation.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        loop = tqdm(loader, desc = f"Epoch {epoch_idx} [Eval]", leave = False)
        for inputs, targets in loop:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            loop.set_postfix(loss = loss.item(), acc = 100. * correct / total)
            
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def main():
    args = args_parser()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.save_dir, "train_log.txt"))
        ]
    )

    setup_seed(args.seed)
    device = get_device()
    logger.info(f"Using device: {device}")
    logger.info(f"Hyperparameters: {vars(args)}")

    # Data Preparation
    train_loader, test_loader, classes = get_data_loaders(
        data_root = args.data_dir, 
        batch_size = args.batch_size
    )
    logger.info(f"Classes: {classes}")
    
    # Model Setup
    model = SimpleCNN(num_classes = 10).to(device)
    log_model_info(model) # Use utility to log structure
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr = args.lr, 
        weight_decay = 1e-4
    )
    
    # Training Loop
    logger.info("Starting training...")
    best_acc = 0.0
    
    with Timer() as t:
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model = model,
                loader = train_loader,
                criterion = criterion,
                optimizer = optimizer,
                device = device,
                epoch_idx = epoch
            )
            
            val_loss, val_acc = evaluate(
                model = model,
                loader = test_loader,
                criterion = criterion,
                device = device,
                epoch_idx = epoch
            )
            
            logger.info(f"Epoch {epoch}/{args.epochs} - "
                        f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                        f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                save_path = os.path.join(args.save_dir, "best_model.pth")
                torch.save(model.state_dict(), save_path)
                logger.info(f"New best model saved with Acc: {best_acc:.2f}%")
    
        logger.info(f"Training completed, Best Acc: {best_acc:.2f}%")

if __name__ == "__main__":
    main()