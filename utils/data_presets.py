"""
Centralized data presets for chapter training and inference scripts.
"""

CIFAR10_MEAN = (
    0.4914,
    0.4822,
    0.4465
)
CIFAR10_STD = (
    0.2023,
    0.1994,
    0.2010
)
CIFAR10_STATS = (CIFAR10_MEAN, CIFAR10_STD)
CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

STL10_MEAN = (
    0.4467,
    0.4398,
    0.4066
)
STL10_STD = (
    0.2603,
    0.2566,
    0.2713
)
STL10_STATS = (STL10_MEAN, STL10_STD)
STL10_CLASSES = [
    "airplane",
    "bird",
    "car",
    "cat",
    "deer",
    "dog",
    "horse",
    "monkey",
    "ship",
    "truck"
]
