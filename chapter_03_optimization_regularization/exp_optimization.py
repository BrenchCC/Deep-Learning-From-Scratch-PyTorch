import os
import sys
import logging

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim 

sys.path.append(os.getcwd())

# Fallback utilities
try:
    from utils import setup_seed
except ImportError:
    def setup_seed(seed): torch.manual_seed(seed); np.random.seed(seed)

from chapter_03_optimization_regularization.optimizers import StochasticGradientDescent, AdaptiveMomentEstimationW

logger = logging.getLogger("Exp_Optimization")

def rosenbrock(x, y):
    """
    The 'Banana Function'. Global min at (1, 1).
    """
    return (1 - x)**2 + 100 * (y - x**2)**2

def run_optimizer(opt_class, name, steps = 1000, lr = 0.001, **kwargs):
    # Fixed starting point (-1.5, -1.0)
    x_val = -1.5
    y_val = -1.0
    params = torch.tensor([x_val, y_val], requires_grad = True)
    
    # Instantiate optimizer (works for both our class and torch.optim class)
    optimizer = opt_class([params], lr = lr, **kwargs)
    
    path = []
    path.append(params.detach().numpy().copy())
    
    for _ in range(steps):
        optimizer.zero_grad()
        loss = rosenbrock(params[0], params[1])
        loss.backward()
        
        # Stability: Gradient Clipping (PyTorch API)
        torch.nn.utils.clip_grad_norm_([params], max_norm = 1.0)
        
        optimizer.step()
        path.append(params.detach().numpy().copy())
        
    return np.array(path)

def main():
    logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    setup_seed(42)
    
    steps = 1500
    
    logger.info("Running Optimization Experiments...")
    
    # 1. Manual SGD w/ Momentum
    path_manual_sgd = run_optimizer(StochasticGradientDescent, "Manual_SGD", steps, lr=0.002, momentum=0.9)
    
    # 2. PyTorch Official SGD w/ Momentum (Comparison)
    path_torch_sgd = run_optimizer(optim.SGD, "Torch_SGD", steps, lr=0.002, momentum=0.9)
    
    # 3. Manual AdamW
    path_manual_adam = run_optimizer(AdaptiveMomentEstimationW, "Manual_AdamW", steps, lr=0.05, weight_decay=0.01)
    
    # Visualization
    logger.info("Plotting results...")
    x = np.linspace(-2, 2.5, 200)
    y = np.linspace(-2, 3, 200)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock(X, Y)
    
    plt.figure(figsize = (12, 8))
    plt.contourf(X, Y, np.log(Z + 1), levels = 30, cmap = 'gray', alpha = 0.3)
    
    # Plot Manual SGD
    plt.plot(path_manual_sgd[:, 0], path_manual_sgd[:, 1], 'r--', linewidth=2, label = 'Manual SGD+Mom')
    # Plot Torch SGD (Should overlap or be very close)
    plt.plot(path_torch_sgd[:, 0], path_torch_sgd[:, 1], 'g:', linewidth=2, label = 'PyTorch SGD+Mom')
    # Plot Manual AdamW
    plt.plot(path_manual_adam[:, 0], path_manual_adam[:, 1], 'b-', linewidth=2, label = 'Manual AdamW')
    
    plt.plot(1, 1, 'k*', markersize = 15, label = 'Global Min')
    plt.plot(-1.5, -1.0, 'ko', label = 'Start')
    
    plt.legend()
    plt.title("Rosenbrock Optimization: Manual Implementation vs PyTorch API")
    plt.savefig("chapter_03_optimization_regularization/images/optimizer_comparison.png")
    logger.info("Saved optimizer_comparison.png")

if __name__ == "__main__":
    main()