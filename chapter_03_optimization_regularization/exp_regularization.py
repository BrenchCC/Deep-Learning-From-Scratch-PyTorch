import os
import sys
import logging

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())

# Fallback utilities
try:
    from utils import setup_seed
except ImportError:
    def setup_seed(seed): torch.manual_seed(seed); np.random.seed(seed)

from chapter_03_optimization_regularization.optimizers import StochasticGradientDescent
from chapter_03_optimization_regularization.regularization import compute_l1_loss

logger = logging.getLogger("Exp_Regularization")

def generate_sparse_data(n_samples = 100, n_features = 50, n_informative = 5):
    """
    Generate regression data where only a few features are informative.
    True weights: [1, 1, 1, 1, 1, 0, 0, ...]
    """
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    true_w = np.zeros(n_features)
    true_w[:n_informative] = 1.0 # Only first 5 features matter
    
    # y = Xw + noise
    y = np.dot(X, true_w) + np.random.normal(0, 0.1, size = n_samples)
    
    return torch.tensor(X, dtype = torch.float32), torch.tensor(y, dtype = torch.float32).unsqueeze(1)

def train_regression(X, y, reg_type = 'None', lambda_reg = 0.01):
    model = nn.Linear(X.shape[1], 1, bias = False)
    # Init with small noise
    nn.init.normal_(model.weight, mean = 0, std = 0.01)
    
    # Configure Optimizer
    # For L2, we use weight_decay inside SGD
    wd = lambda_reg if reg_type == 'L2' else 0.0
    optimizer = StochasticGradientDescent(model.parameters(), lr = 0.1, weight_decay = wd)
    
    for epoch in range(500):
        optimizer.zero_grad()
        output = model(X)
        loss = nn.functional.mse_loss(output, y)
        
        # For L1, we add it explicitly to loss
        if reg_type == 'L1':
            loss += compute_l1_loss(model, lambda_reg)
            
        loss.backward()
        optimizer.step()
        
    return model.weight.detach().numpy().flatten()

def plot_weights(w_l1, w_l2, w_none):
    plt.figure(figsize = (12, 6))
    
    # Plot bars
    indices = np.arange(len(w_l1))
    width = 0.25
    
    plt.bar(indices - width, w_none, width, label = 'No Reg', color = 'gray', alpha = 0.5)
    plt.bar(indices, w_l2, width, label = 'L2 (Ridge)', color = 'blue', alpha = 0.7)
    plt.bar(indices + width, w_l1, width, label = 'L1 (Lasso)', color = 'red', alpha = 0.7)
    
    plt.axhline(0, color = 'black', linewidth = 0.8)
    plt.ylabel("Weight Value")
    plt.xlabel("Feature Index")
    plt.title("L1 vs L2 Regularization Effect on Weights")
    plt.legend()
    
    # Highlight the informative features
    plt.axvline(4.5, color = 'green', linestyle = '--', label = 'Informative Boundary')
    
    plt.savefig("chapter_03_optimization_regularization/images/regularization_comparison.png")
    logger.info("Saved regularization_comparison.png")

def main():
    logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    setup_seed(42)
    
    X, y = generate_sparse_data()
    logger.info("Data Generated. Features: 50, Informative: 5")
    
    # 1. No Regularization
    w_none = train_regression(X, y, reg_type = 'None')
    logger.info(f"No Reg - Zero Weights: {np.sum(np.abs(w_none) < 1e-3)}")
    
    # 2. L2 Regularization (Weight Decay)
    w_l2 = train_regression(X, y, reg_type = 'L2', lambda_reg = 0.1)
    logger.info(f"L2 Reg - Zero Weights: {np.sum(np.abs(w_l2) < 1e-3)} (Weights shrink but don't vanish)")
    
    # 3. L1 Regularization
    w_l1 = train_regression(X, y, reg_type = 'L1', lambda_reg = 0.05)
    logger.info(f"L1 Reg - Zero Weights: {np.sum(np.abs(w_l1) < 1e-3)} (Sparsity achieved!)")
    
    plot_weights(w_l1, w_l2, w_none)

if __name__ == "__main__":
    main()