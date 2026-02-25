import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

sys.path.append(os.getcwd())

from chapter_03_optimization_regularization.exp_common import setup_experiment, log_section, ensure_parent_dir
from chapter_03_optimization_regularization.optimizers import StochasticGradientDescent, AdaptiveMomentEstimationW

def rosenbrock(x, y):
    """
    The Rosenbrock function (banana function).
    Global minimum is at (1, 1).

    Args:
        x (torch.Tensor | np.ndarray): x coordinate.
        y (torch.Tensor | np.ndarray): y coordinate.

    Returns:
        torch.Tensor | np.ndarray: Function value.
    """
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

def run_optimizer(opt_class, steps = 1000, lr = 0.001, **kwargs):
    """
    Run one optimizer on Rosenbrock function and return parameter path.

    Args:
        opt_class (type): Optimizer class.
        steps (int): Number of optimization steps.
        lr (float): Learning rate.
        **kwargs: Extra optimizer arguments.

    Returns:
        np.ndarray: Optimization path with shape (steps + 1, 2).
    """
    params = torch.tensor([-1.5, -1.0], requires_grad = True)
    optimizer = opt_class([params], lr = lr, **kwargs)

    path = [params.detach().numpy().copy()]
    for _ in range(steps):
        optimizer.zero_grad()
        loss = rosenbrock(params[0], params[1])
        loss.backward()
        torch.nn.utils.clip_grad_norm_([params], max_norm = 1.0)
        optimizer.step()
        path.append(params.detach().numpy().copy())

    return np.array(path)

def plot_paths(path_manual_sgd, path_torch_sgd, path_manual_adam, save_path: str):
    """
    Plot optimizer trajectories on Rosenbrock contour map.

    Args:
        path_manual_sgd (np.ndarray): Path of custom SGD.
        path_torch_sgd (np.ndarray): Path of torch SGD.
        path_manual_adam (np.ndarray): Path of custom AdamW.
        save_path (str): Output figure path.
    """
    x = np.linspace(-2, 2.5, 200)
    y = np.linspace(-2, 3, 200)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock(X, Y)

    plt.figure(figsize = (12, 8))
    plt.contourf(X, Y, np.log(Z + 1), levels = 30, cmap = 'gray', alpha = 0.3)
    plt.plot(path_manual_sgd[:, 0], path_manual_sgd[:, 1], 'r--', linewidth = 2, label = 'Manual SGD+Mom')
    plt.plot(path_torch_sgd[:, 0], path_torch_sgd[:, 1], 'g:', linewidth = 2, label = 'PyTorch SGD+Mom')
    plt.plot(path_manual_adam[:, 0], path_manual_adam[:, 1], 'b-', linewidth = 2, label = 'Manual AdamW')
    plt.plot(1, 1, 'k*', markersize = 15, label = 'Global Min')
    plt.plot(-1.5, -1.0, 'ko', label = 'Start')
    plt.legend()
    plt.title("Rosenbrock Optimization: Manual Implementation vs PyTorch API")
    ensure_parent_dir(save_path)
    plt.savefig(save_path)
    plt.close()

def main():
    logger = setup_experiment(logger_name = "Exp_Optimization", seed = 42)
    steps = 1500
    output_path = "chapter_03_optimization_regularization/images/optimizer_comparison.png"

    log_section(logger, "Step 1/3: Run Optimization Paths")
    path_manual_sgd = run_optimizer(
        StochasticGradientDescent,
        steps = steps,
        lr = 0.002,
        momentum = 0.9
    )
    path_torch_sgd = run_optimizer(
        optim.SGD,
        steps = steps,
        lr = 0.002,
        momentum = 0.9
    )
    path_manual_adam = run_optimizer(
        AdaptiveMomentEstimationW,
        steps = steps,
        lr = 0.05,
        weight_decay = 0.01
    )

    log_section(logger, "Step 2/3: Plot and Save Figure")
    plot_paths(
        path_manual_sgd = path_manual_sgd,
        path_torch_sgd = path_torch_sgd,
        path_manual_adam = path_manual_adam,
        save_path = output_path
    )

    log_section(logger, "Step 3/3: Summary")
    logger.info(f"Manual SGD final point: {path_manual_sgd[-1]}")
    logger.info(f"Torch SGD final point: {path_torch_sgd[-1]}")
    logger.info(f"Manual AdamW final point: {path_manual_adam[-1]}")
    logger.info(f"Saved figure: {output_path}")

if __name__ == "__main__":
    main()
