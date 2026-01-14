"""
src/dataset.py
Handles STL-10 data loading and preprocessing.
Includes strong augmentation for the small labeled training set (5k images).
"""

import os
import logging
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

logger = logging.getLogger("Dataset Pipeline")

def get_stl10_loaders(data_root = "./data", batch_size = 64, num_workers = 2):
    """
    Returns training and validation DataLoaders for STL-10.
    
    Args:
        data_root (str): Path to store dataset.
        batch_size (int): Batch size.
        num_workers (int): Number of subprocesses for data loading.
        
    Returns:
        train_loader, test_loader, classes (list)
    """
    # STL-10 Statistics (calculated on the labeled set)
    mean = [0.4467, 0.4398, 0.4066]
    std  = [0.2603, 0.2566, 0.2713]

    # --- Training Transforms (Augmentation) ---
    # 1. Pad by 4 pixels then random crop back to 96x96 (Simulates shift)
    # 2. Random horizontal flip
    # 3. Tensor conversion & Normalization
    train_transform = transforms.Compose([
        transforms.RandomCrop(96, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # --- Test Transforms (Clean) ---
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # --- Datasets ---
    # STL-10 has a 'train' split (5k labeled) and 'test' split (8k labeled).
    # It also has 'unlabeled' (100k), which we skip for this supervised task.
    train_set = torchvision.datasets.STL10(
        root = data_root, 
        split = 'train', 
        download = True, 
        transform = train_transform
    )

    test_set = torchvision.datasets.STL10(
        root = data_root, 
        split = 'test', 
        download = True, 
        transform = test_transform
    )

    # --- Loaders ---
    # MPS (Mac) optimization: pin_memory=True usually helps data transfer speed
    train_loader = DataLoader(
        train_set, 
        batch_size = batch_size, 
        shuffle = True, 
        num_workers = num_workers, 
        pin_memory = True
    )

    test_loader = DataLoader(
        test_set, 
        batch_size = batch_size, 
        shuffle = False, 
        num_workers = num_workers, 
        pin_memory = True
    )

    classes = train_set.classes  # ['airplane', 'bird', 'car', ...]

    return train_loader, test_loader, classes