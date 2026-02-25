import os
import sys

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())

from chapter_03_optimization_regularization.exp_common import setup_experiment, log_section, ensure_parent_dir
from chapter_03_optimization_regularization.optimizers import StochasticGradientDescent
from chapter_03_optimization_regularization.regularization import compute_l1_loss

def generate_sparse_data(n_samples = 100, n_features = 50, n_informative = 5):
    """
    Generate sparse linear regression data.

    Args:
        n_samples (int): Number of samples.
        n_features (int): Total number of features.
        n_informative (int): Number of true informative features.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Input matrix X and target y.
    """
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    true_w = np.zeros(n_features)
    true_w[:n_informative] = 1.0
    y = np.dot(X, true_w) + np.random.normal(0, 0.1, size = n_samples)

    return (
        torch.tensor(X, dtype = torch.float32),
        torch.tensor(y, dtype = torch.float32).unsqueeze(1)
    )

def train_regression(X, y, reg_type = 'None', lambda_reg = 0.01):
    """
    Train linear regression with optional L1/L2 regularization.

    Args:
        X (torch.Tensor): Feature matrix.
        y (torch.Tensor): Target tensor.
        reg_type (str): 'None', 'L1', or 'L2'.
        lambda_reg (float): Regularization coefficient.

    Returns:
        np.ndarray: Learned weight vector.
    """
    model = nn.Linear(X.shape[1], 1, bias = False)
    nn.init.normal_(model.weight, mean = 0, std = 0.01)

    weight_decay = lambda_reg if reg_type == 'L2' else 0.0
    optimizer = StochasticGradientDescent(model.parameters(), lr = 0.1, weight_decay = weight_decay)

    for _ in range(500):
        optimizer.zero_grad()
        output = model(X)
        loss = nn.functional.mse_loss(output, y)
        if reg_type == 'L1':
            loss = loss + compute_l1_loss(model, lambda_reg)
        loss.backward()
        optimizer.step()

    return model.weight.detach().numpy().flatten()

def plot_weights(w_l1, w_l2, w_none, save_path: str):
    """
    Plot learned weights for three regularization settings.

    Args:
        w_l1 (np.ndarray): Weights from L1 run.
        w_l2 (np.ndarray): Weights from L2 run.
        w_none (np.ndarray): Weights from no regularization run.
        save_path (str): Output figure path.
    """
    plt.figure(figsize = (12, 6))
    indices = np.arange(len(w_l1))
    width = 0.25

    plt.bar(indices - width, w_none, width, label = 'No Reg', color = 'gray', alpha = 0.5)
    plt.bar(indices, w_l2, width, label = 'L2 (Ridge)', color = 'blue', alpha = 0.7)
    plt.bar(indices + width, w_l1, width, label = 'L1 (Lasso)', color = 'red', alpha = 0.7)
    plt.axhline(0, color = 'black', linewidth = 0.8)
    plt.axvline(4.5, color = 'green', linestyle = '--', label = 'Informative Boundary')

    plt.ylabel("Weight Value")
    plt.xlabel("Feature Index")
    plt.title("L1 vs L2 Regularization Effect on Weights")
    plt.legend()

    ensure_parent_dir(save_path)
    plt.savefig(save_path)
    plt.close()

def main():
    logger = setup_experiment(logger_name = "Exp_Regularization", seed = 42)
    output_path = "chapter_03_optimization_regularization/images/regularization_comparison.png"

    log_section(logger, "Step 1/3: Generate Data")
    X, y = generate_sparse_data()
    logger.info("Generated sparse data: n_features = 50, n_informative = 5")

    log_section(logger, "Step 2/3: Train Under Different Regularization Settings")
    w_none = train_regression(X, y, reg_type = 'None')
    w_l2 = train_regression(X, y, reg_type = 'L2', lambda_reg = 0.1)
    w_l1 = train_regression(X, y, reg_type = 'L1', lambda_reg = 0.05)

    logger.info(f"No Reg   - near-zero weights: {np.sum(np.abs(w_none) < 1e-3)}")
    logger.info(f"L2 Reg   - near-zero weights: {np.sum(np.abs(w_l2) < 1e-3)}")
    logger.info(f"L1 Reg   - near-zero weights: {np.sum(np.abs(w_l1) < 1e-3)}")

    log_section(logger, "Step 3/3: Plot and Save Figure")
    plot_weights(
        w_l1 = w_l1,
        w_l2 = w_l2,
        w_none = w_none,
        save_path = output_path
    )
    logger.info(f"Saved figure: {output_path}")

if __name__ == "__main__":
    main()
