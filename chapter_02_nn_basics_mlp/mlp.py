import os
import sys
import logging
import argparse

import numpy as np
import matplotlib.pyplot as plt
# New import for 3D plotting
from mpl_toolkits.mplot3d import Axes3D

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

sys.path.append(os.getcwd())

try:
    from utils import get_device, setup_seed, Timer, log_model_info
except ImportError:
    # Fallback if utils not present
    def get_device(): return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    def setup_seed(seed): torch.manual_seed(seed); np.random.seed(seed)
    class Timer: 
        def __enter__(self): pass 
        def __exit__(self, *args): pass
    def log_model_info(model): pass

# 1. Global Logger Configuration
logger = logging.getLogger("MLP-Demo-MultiDim")

class MLP(nn.Module):
    """
    A Multi-Layer Perceptron (MLP) demonstrating Universal Approximation.
    Uses ReLU activation to approximate non-linear functions.
    """

    def __init__(
        self, 
        input_dim: int = 1,
        hidden_dim: int = 100,
        output_dim: int = 1
    ):
        """
        Initialize the MLP model.

        Args:
            input_dim (int): Input feature dimension.
            hidden_dim (int): Hidden layer width.
            output_dim (int): Output dimension.
        """
        super().__init__()
        # Strictly adhering to spacing preference
        self.mlp_net = nn.Sequential(
            nn.Linear(in_features = input_dim, out_features = hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features = hidden_dim, out_features = output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        """
        return self.mlp_net(x)

def generate_sine_data(
    start: float,
    end: float,
    n_samples: int, 
    noise_std: float = 0.1,
    save_path: str = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    [Original Function Preserved]
    Generate synthetic data for Sine wave regression (1D -> 1D).
    y = sin(x) + noise
    """
    logger.info(f"Generating 1D data: Range[{start:.2f}, {end:.2f}], Samples={n_samples}")

    x = np.linspace(
        start = start,
        stop = end,
        num = n_samples
    )
    y = np.sin(x)

    if noise_std > 0:
        noise = np.random.normal(
            loc = 0.0,
            scale = noise_std,
            size = n_samples
        )
        y = y + noise

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok = True)
        with open(save_path, "w") as f:
            for i in range(n_samples):
                f.write(f"{x[i]:.6f}\t{y[i]:.6f}\n")
        logger.info(f"Data saved to {save_path}")

    x_tensor = torch.from_numpy(x).float().unsqueeze(1)
    y_tensor = torch.from_numpy(y).float().unsqueeze(1)

    return x_tensor, y_tensor

def generate_surface_data(
    n_samples: int,
    range_min: float = -3.0,
    range_max: float = 3.0,
    noise_std: float = 0.1,
    save_path: str = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    [New Function]
    Generate synthetic data for 3D Surface regression (2D -> 1D).
    z = sin(x) + cos(y) + noise

    Args:
        n_samples (int): Number of points (total will be roughly n_samples, sampled randomly).
        range_min (float): Min value for x and y.
        range_max (float): Max value for x and y.
        noise_std (float): Noise level.
        save_path (str, optional): Save path.
    """
    logger.info(f"Generating 2D->1D data: Range[{range_min}, {range_max}], Samples={n_samples}")

    # Random sampling in 2D space
    x = np.random.uniform(low = range_min, high = range_max, size = n_samples)
    y = np.random.uniform(low = range_min, high = range_max, size = n_samples)
    
    # Function: z = sin(x) + cos(y)
    z = np.sin(x) + np.cos(y)

    if noise_std > 0:
        noise = np.random.normal(loc = 0.0, scale = noise_std, size = n_samples)
        z = z + noise

    # Stack x and y for input [n_samples, 2]
    inputs = np.stack((x, y), axis = 1)
    targets = z

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok = True)
        np.savetxt(fname = save_path, X = np.hstack((inputs, targets.reshape(-1, 1))), fmt = '%.6f')
        logger.info(f"Data saved to {save_path}")

    inputs_tensor = torch.from_numpy(inputs).float()
    targets_tensor = torch.from_numpy(targets).float().unsqueeze(1)

    return inputs_tensor, targets_tensor

def train_model(
    model: MLP,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int,
    lr: float,
    device: torch.device
):
    """
    [Original Function Preserved]
    Execute the training loop.
    """
    model.to(device)
    x_train = x_train.to(device)
    y_train = y_train.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        params = model.parameters(), 
        lr = lr
    )    

    logger.info(f"Starting training: Epochs={epochs}, LR={lr}, Device={device}")

    pbar = tqdm(range(epochs), desc = "Training Progress")

    with Timer():
        for epoch in pbar:
            # Forward
            y_pred = model(x_train)
            loss = criterion(input = y_pred, target = y_train)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update tqdm
            pbar.set_postfix(loss = f"{loss.item():.6f}")

            # Periodic logging
            if (epoch + 1) % (epochs // 10) == 0:
                logger.info(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}")

def visualize_results(
    model: MLP, 
    x_train: torch.Tensor, 
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    mode: str,
    save_path: str,
    device: torch.device
):
    """
    [Original Function Preserved - Handles 1D Visualization]
    Visualize prediction results for 1D input. 
    """
    model.eval()
    
    x_test_sorted, indices = torch.sort(x_test, dim = 0)
    y_test_sorted = y_test[indices].view(x_test_sorted.shape)
    
    with torch.no_grad():
        y_pred = model(x_test_sorted.to(device)).cpu().numpy()

    plt.figure(figsize = (12, 7), dpi = 150)

    # 1. Plot Ground Truth
    plt.plot(
        x_test_sorted.numpy(), 
        y_test_sorted.numpy(), 
        label = "Ground Truth", 
        color = 'gray', 
        linestyle = '--', 
        linewidth = 2,
        alpha = 0.5
    )

    # 2. Plot Training Data
    plt.scatter(
        x_train.numpy(), 
        y_train.numpy(), 
        label = 'Training Data', 
        color = 'blue', 
        s = 10, 
        alpha = 0.5
    )

    # 3. Plot Model Prediction
    plt.plot(
        x_test_sorted.numpy(), 
        y_pred, 
        label = 'Model Prediction', 
        color = 'red', 
        linewidth = 2
    )

    plt.title(label = f"MLP 1D: {mode.capitalize()} Mode")
    plt.legend()
    plt.grid(visible = True, alpha = 0.3)
    
    os.makedirs(os.path.dirname(save_path), exist_ok = True)
    plt.savefig(fname = save_path)
    logger.info(f"Visualization saved to {save_path}")
    plt.close()

def visualize_3d_results(
    model: MLP,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    save_path: str,
    device: torch.device
):
    """
    [New Function]
    Visualize prediction results for 2D input using 3D surface plot.
    """
    model.eval()
    logger.info("Generating 3D surface mesh for visualization...")

    # Create a meshgrid for plotting the surface
    range_val = 3.2
    grid_x = np.linspace(-range_val, range_val, 100)
    grid_y = np.linspace(-range_val, range_val, 100)
    X, Y = np.meshgrid(grid_x, grid_y)
    
    # Flatten for model inference
    # input shape: [10000, 2]
    flat_inputs = np.stack((X.ravel(), Y.ravel()), axis = 1) 
    inputs_tensor = torch.from_numpy(flat_inputs).float().to(device)

    with torch.no_grad():
        Z_pred = model(inputs_tensor).cpu().numpy().reshape(X.shape)
        # Calculate Ground Truth for comparison (optional)
        Z_true = np.sin(X) + np.cos(Y)

    # 3D Plotting
    fig = plt.figure(figsize = (14, 8), dpi = 150)
    
    # Subplot 1: Model Prediction Surface with Train Points
    ax = fig.add_subplot(1, 1, 1, projection = '3d')
    
    # Plot Prediction Surface
    surf = ax.plot_surface(
        X, Y, Z_pred, 
        cmap = 'viridis', 
        alpha = 0.8, 
        edgecolor = 'none'
    )
    
    # Plot Training Data Points (Scatter)
    # Only plot a subset to avoid clutter
    subset_idx = np.random.choice(x_train.shape[0], size = min(200, x_train.shape[0]), replace = False)
    ax.scatter(
        x_train[subset_idx, 0].numpy(),
        x_train[subset_idx, 1].numpy(),
        y_train[subset_idx, 0].numpy(),
        c = 'red',
        marker = 'o',
        s = 20,
        label = 'Train Data'
    )

    ax.set_title(label = "MLP 3D Surface Fitting: z = sin(x) + cos(y)")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    fig.colorbar(surf, ax = ax, shrink = 0.5, aspect = 5)

    os.makedirs(os.path.dirname(save_path), exist_ok = True)
    plt.savefig(fname = save_path)
    logger.info(f"3D Visualization saved to {save_path}")
    plt.close()

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description = "MLP Universal Approximation Demo (1D & 2D)")
    
    # Input Dimension Control
    parser.add_argument(
        "--input_dim", 
        type = int, 
        default = 1, 
        choices = [1, 2],
        help = "Input dimension: 1 for Sine wave, 2 for 3D Surface"
    )

    # Experiment Mode
    parser.add_argument(
        "--mode", 
        type = str, 
        default = "standard", 
        choices = ["standard", "extrapolate"],
        help = "Mode: 'standard' for interpolation, 'extrapolate' for unseen range testing"
    )
    
    # Training Hyperparameters
    parser.add_argument("--epochs", type = int, default = 2000, help = "Number of training epochs")
    parser.add_argument("--lr", type = float, default = 0.01, help = "Learning rate")
    parser.add_argument("--hidden_dim", type = int, default = 400, help = "Hidden layer width")
    parser.add_argument("--n_samples", type = int, default = 2000, help = "Number of training samples") # Increased default
    
    # I/O Config
    parser.add_argument("--save_dir", type = str, default = "chapter_02_nn_basics_mlp", help = "Directory to save outputs")
    parser.add_argument("--seed", type = int, default = 42, help = "Random seed")

    return parser.parse_args()

def mlp_demo():
    """
    Main execution pipeline.
    """
    # 1. Logging Config (Strictly in main)
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()],
        datefmt = '%Y-%m-%d %H:%M:%S'
    )
    
    # 2. Argument Parsing
    args = parse_args()
    logger.info(f"Arguments parsed: {vars(args)}")

    # 3. Setup
    setup_seed(seed = args.seed)
    device = get_device()
    
    # 4. Data Generation Strategy based on Input Dim
    if args.input_dim == 1:
        # --- 1D Logic (Original) ---
        if args.mode == "standard":
            train_start, train_end = -np.pi, np.pi
            test_start, test_end = -np.pi, np.pi
        elif args.mode == "extrapolate":
            train_start, train_end = -np.pi, 0.0
            test_start, test_end = -np.pi, np.pi

        x_train, y_train = generate_sine_data(
            start = train_start,
            end = train_end,
            n_samples = args.n_samples,
            save_path = os.path.join(args.save_dir, "data", f"1d_{args.mode}_train.txt")
        )
        # Test data for 1D
        x_test, y_test = generate_sine_data(
            start = test_start,
            end = test_end,
            n_samples = 300, 
            noise_std = 0.0,
            save_path = None
        )

    elif args.input_dim == 2:
        # --- 2D Logic (New) ---
        # Note: 'extrapolate' logic for 2D is simplified here to just random sampling range
        x_train, y_train = generate_surface_data(
            n_samples = args.n_samples,
            range_min = -3.0,
            range_max = 3.0,
            noise_std = 0.1,
            save_path = os.path.join(args.save_dir, "data", "2d_surface_train.txt")
        )
        # x_test/y_test not explicitly needed for 3D vis function as it generates its own mesh
        x_test, y_test = None, None

    # 5. Model Initialization (Dynamic Input Dim)
    model = MLP(
        input_dim = args.input_dim,
        hidden_dim = args.hidden_dim,
        output_dim = 1
    )
    log_model_info(model = model)

    # 6. Training
    train_model(
        model = model,
        x_train = x_train,
        y_train = y_train,
        epochs = args.epochs,
        lr = args.lr,
        device = device
    )

    # 7. Visualization & Saving
    if args.input_dim == 1:
        vis_path = os.path.join(args.save_dir, "images", f"mlp_1d_{args.mode}_result.png")
        visualize_results(
            model = model,
            x_train = x_train,
            y_train = y_train,
            x_test = x_test,
            y_test = y_test,
            mode = args.mode,
            save_path = vis_path,
            device = device
        )
    elif args.input_dim == 2:
        vis_path = os.path.join(args.save_dir, "images", "mlp_2d_surface_result.png")
        visualize_3d_results(
            model = model,
            x_train = x_train,
            y_train = y_train,
            save_path = vis_path,
            device = device
        )

    # Save Model Checkpoint
    model_path = os.path.join(args.save_dir, "models", f"mlp_{args.input_dim}d.pth")
    os.makedirs(os.path.dirname(model_path), exist_ok = True)
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    mlp_demo()