"""
Defines the ResNet architecture from scratch with a toggle for residual connections.
This allows strictly comparing 'Plain Net' vs 'ResNet' to visualize the degradation problem.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: Implement the BasicBlock and ResNet classes here.