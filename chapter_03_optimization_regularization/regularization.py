import os
import sys
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.getcwd())
logger = logging.getLogger("Regularization")

class LabelSmoothingLoss(nn.Module):
    """
    Cross Entropy Loss with Label Smoothing.
    Theory: y_ls = (1 - epsilon) * y_hot + epsilon / K
    """
    def __init__(self, epsilon: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: (Batch, Num_Classes)
        # target: (Batch) - Class indices
        num_classes = predictions.size(-1)
        log_preds = F.log_softmax(predictions, dim = -1)
        
        # Calculate Cross Entropy with Hard Targets (Standard)
        # loss = - sum(y_hot * log_p)
        loss_nll = F.nll_loss(log_preds, target, reduction = self.reduction)
        
        # Calculate Entropy with Uniform Distribution (Smoothing part)
        # loss = - sum(1/K * log_p) = - mean(log_p)
        loss_smooth = -log_preds.mean(dim = -1)
        
        if self.reduction == 'mean':
            loss_smooth = loss_smooth.mean()
        elif self.reduction == 'sum':
            loss_smooth = loss_smooth.sum()
            
        # Combine
        return (1 - self.epsilon) * loss_nll + self.epsilon * loss_smooth        

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (applied in main path of residual blocks).
    """
    if drop_prob == 0. or not training:
        return x
    
    keep_prob = 1 - drop_prob
    # Work with any number of dimensions, handling broadcasting
    # shape: (Batch, 1, 1, ...)
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    
    # Generate random tensor from Bernoulli distribution
    random_tensor = keep_prob + torch.rand(shape, device = x.device)
    random_tensor.floor_()  # binarize to 0 or 1
    
    # Scale output to maintain expected value
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """
    DropPath module for usage in nn.Sequential or layers.
    """
    def __init__(self, drop_prob = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def compute_l1_loss(model, lambda_l1):
    """
    Manually compute L1 regularization loss (Lasso).
    """
    l1_loss = 0.0
    for param in model.parameters():
        l1_loss += torch.sum(torch.abs(param))
    return lambda_l1 * l1_loss


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()],
        datefmt = '%Y-%m-%d %H:%M:%S'
    )

    # 1. Test Label Smoothing
    logits = torch.tensor([[10.0, 2.0, 1.0], [1.0, 5.0, 1.0]]) # 2 samples, 3 classes
    targets = torch.tensor([0, 1])
    
    crit_smooth = LabelSmoothingLoss(epsilon = 0.1)
    loss = crit_smooth(logits, targets)
    logger.info(f"Label Smoothing Loss: {loss.item():.4f}")
    
    # 2. Test Drop Path
    x = torch.ones(4, 2) # Batch=4
    out = drop_path(x, drop_prob = 0.5, training = True)
    logger.info(f"Drop Path Output (Should have zeros and twos):\n{out}")
