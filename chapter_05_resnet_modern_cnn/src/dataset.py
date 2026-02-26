"""
STL-10 data loading and preprocessing helpers.
"""

import os
import sys
import logging

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())

from utils import STL10_STATS

logger = logging.getLogger("STL10Dataset")


def get_stl10_loaders(data_root = "./data", batch_size = 64, num_workers = 2):
    """
    Build train and test DataLoaders for STL-10.

    Args:
        data_root (str): Dataset root directory.
        batch_size (int): Batch size.
        num_workers (int): Number of DataLoader workers.

    Returns:
        tuple: (train_loader, test_loader, classes)
    """
    train_transform = transforms.Compose([
        transforms.RandomCrop(96, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*STL10_STATS)
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*STL10_STATS)
    ])

    train_set = torchvision.datasets.STL10(
        root = data_root,
        split = "train",
        download = True,
        transform = train_transform
    )
    test_set = torchvision.datasets.STL10(
        root = data_root,
        split = "test",
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

    logger.info(f"Loaded STL-10 | Train: {len(train_set)} | Test: {len(test_set)}")
    return train_loader, test_loader, train_set.classes
