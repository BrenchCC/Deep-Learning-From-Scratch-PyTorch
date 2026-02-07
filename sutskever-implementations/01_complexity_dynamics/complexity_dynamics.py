import os
import sys
import logging
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

sys.path.append(os.getcwd())

from utils import set_seed
logger = logging.getLogger("ComplexityDynamics")


def parse_args():
    """Parse command-line arguments.

    Args:
        None: This function uses argparse to read CLI arguments.

    Returns:
        argparse.Namespace: Parsed arguments including sizes, steps, and output path.
    """
    parser = argparse.ArgumentParser(description = "Complexity dynamics demos")
    parser.add_argument(
        "--seed",
        type = int,
        default = 42,
        help = "Random seed for reproducibility"
    )
    parser.add_argument(
        "--ca-size",
        type = int,
        default = 100,
        help = "Cellular automaton width"
    )
    parser.add_argument(
        "--ca-steps",
        type = int,
        default = 100,
        help = "Cellular automaton time steps"
    )
    parser.add_argument(
        "--diffusion-size",
        type = int,
        default = 50,
        help = "Diffusion grid size"
    )
    parser.add_argument(
        "--diffusion-steps",
        type = int,
        default = 50,
        help = "Diffusion time steps"
    )
    parser.add_argument(
        "--diffusion-rate",
        type = float,
        default = 0.2,
        help = "Diffusion rate"
    )
    parser.add_argument(
        "--output-dir",
        type = Path,
        default = Path("sutskever-implementations/01_complexity_dynamics/images"),
        help = "Directory to save images"
    )
    parser.add_argument(
        "--no-save",
        action = "store_true",
        help = "Disable saving figures to disk"
    )
    parser.add_argument(
        "--show",
        action = "store_true",
        help = "Show figures in a window"
    )
    return parser.parse_args()


def rule_30(left, center, right):
    """Apply Rule 30 transition.

    Args:
        left (int): Left neighbor state (0 or 1).
        center (int): Center cell state (0 or 1).
        right (int): Right neighbor state (0 or 1).

    Returns:
        int: Next state (0 or 1).
    """
    pattern = (left << 2) | (center << 1) | right
    rule = 30
    return (rule >> pattern) & 1


def evolve_ca(initial_state, steps, rule_func):
    """Evolve a 1D cellular automaton.

    Args:
        initial_state (np.ndarray): Initial 1D state.
        steps (int): Number of time steps.
        rule_func (callable): Transition rule function.

    Returns:
        np.ndarray: Evolution history with shape (steps, size).
    """
    size = len(initial_state)
    history = np.zeros((steps, size), dtype = int)
    history[0] = initial_state

    for t in range(1, steps):
        for i in range(size):
            left = history[t - 1, (i - 1) % size]
            center = history[t - 1, i]
            right = history[t - 1, (i + 1) % size]
            history[t, i] = rule_func(left, center, right)

    return history


def measure_entropy_over_time(history):
    """Compute Shannon entropy at each time step.

    Args:
        history (np.ndarray): Evolution history of the automaton.

    Returns:
        np.ndarray: Entropy values over time.
    """
    entropies = []

    for t in range(len(history)):
        state = history[t]
        unique, counts = np.unique(state, return_counts = True)
        probs = counts / len(state)
        ent = entropy(probs, base = 2)
        entropies.append(ent)

    return np.array(entropies)


def measure_spatial_complexity(history):
    """Measure spatial pattern complexity by counting transitions.

    Args:
        history (np.ndarray): Evolution history of the automaton.

    Returns:
        np.ndarray: Transition counts over time.
    """
    complexities = []

    for t in range(len(history)):
        state = history[t]
        transitions = np.sum(np.abs(np.diff(state)))
        complexities.append(transitions)

    return np.array(complexities)


def diffusion_2d(grid, steps, diffusion_rate = 0.1):
    """Run a simple 2D diffusion simulation.

    Args:
        grid (np.ndarray): Initial 2D grid.
        steps (int): Number of diffusion steps.
        diffusion_rate (float): Mixing rate per step.

    Returns:
        np.ndarray: History of grids over time.
    """
    history = [grid.copy()]

    for _ in range(steps):
        new_grid = grid.copy()
        height, width = grid.shape

        for i in range(1, height - 1):
            for j in range(1, width - 1):
                neighbors = (
                    grid[i - 1, j] + grid[i + 1, j] +
                    grid[i, j - 1] + grid[i, j + 1]
                ) / 4
                new_grid[i, j] = (
                    (1 - diffusion_rate) * grid[i, j] +
                    diffusion_rate * neighbors
                )

        grid = new_grid
        history.append(grid.copy())

    return np.array(history)


def measure_mixing_entropy(history, bins = 20):
    """Compute entropy of diffusion states after discretization.

    Args:
        history (np.ndarray): History of 2D grids.
        bins (int): Number of histogram bins.

    Returns:
        np.ndarray: Entropy values over time.
    """
    entropies = []

    for t in range(len(history)):
        flat = history[t].ravel()
        counts = np.histogram(flat, bins = bins)[0]
        probs = counts[counts > 0] / counts.sum()
        entropies.append(entropy(probs, base = 2))

    return np.array(entropies)


def save_rule_30_figure(evolution, output_path, show, no_save):
    """Save the Rule 30 evolution figure.

    Args:
        evolution (np.ndarray): Evolution history.
        output_path (Path): Output image path.
        show (bool): Whether to display the figure.
        no_save (bool): Whether to skip saving the figure.

    Returns:
        None: The function writes an image to disk.
    """
    plt.figure(figsize = (12, 6))
    plt.imshow(evolution, cmap = "binary", interpolation = "nearest")
    plt.title("Rule 30 Cellular Automaton - Complexity Growth")
    plt.xlabel("Cell Position")
    plt.ylabel("Time Step")
    plt.colorbar(label = "State")
    plt.tight_layout()
    if not no_save:
        plt.savefig(output_path, dpi = 150)
    if show:
        plt.show()
    plt.close()


def save_entropy_complexity_figure(entropies, complexities, output_path, show, no_save):
    """Save the entropy and complexity figure.

    Args:
        entropies (np.ndarray): Entropy over time.
        complexities (np.ndarray): Complexity over time.
        output_path (Path): Output image path.
        show (bool): Whether to display the figure.
        no_save (bool): Whether to skip saving the figure.

    Returns:
        None: The function writes an image to disk.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (14, 4))

    ax1.plot(entropies, linewidth = 2)
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Shannon Entropy (bits)")
    ax1.set_title("Entropy Growth Over Time")
    ax1.grid(True, alpha = 0.3)

    ax2.plot(complexities, linewidth = 2, color = "orange")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Spatial Complexity (transitions)")
    ax2.set_title("Spatial Pattern Complexity")
    ax2.grid(True, alpha = 0.3)

    plt.tight_layout()
    if not no_save:
        fig.savefig(output_path, dpi = 150)
    if show:
        plt.show()
    plt.close(fig)


def save_coffee_grid_figure(mixing_history, output_path, show, no_save):
    """Save the diffusion grid snapshots figure.

    Args:
        mixing_history (np.ndarray): Diffusion history.
        output_path (Path): Output image path.
        show (bool): Whether to display the figure.
        no_save (bool): Whether to skip saving the figure.

    Returns:
        None: The function writes an image to disk.
    """
    fig, axes = plt.subplots(2, 4, figsize = (16, 8))
    timesteps = [0, 5, 10, 15, 20, 30, 40, 50]

    for ax, t in zip(axes.flat, timesteps):
        ax.imshow(mixing_history[t], cmap = "YlOrBr", vmin = 0, vmax = 1)
        ax.set_title(f"Time Step {t}")
        ax.axis("off")

    plt.suptitle("Irreversible Mixing: The Coffee Automaton", fontsize = 14, y = 1.02)
    plt.tight_layout()
    if not no_save:
        fig.savefig(output_path, dpi = 150, bbox_inches = "tight")
    if show:
        plt.show()
    plt.close(fig)


def save_mixing_entropy_figure(mixing_entropies, output_path, show, no_save):
    """Save the mixing entropy curve figure.

    Args:
        mixing_entropies (np.ndarray): Entropy values.
        output_path (Path): Output image path.
        show (bool): Whether to display the figure.
        no_save (bool): Whether to skip saving the figure.

    Returns:
        None: The function writes an image to disk.
    """
    plt.figure(figsize = (10, 5))
    plt.plot(mixing_entropies, linewidth = 2)
    plt.xlabel("Time Step")
    plt.ylabel("Spatial Entropy (bits)")
    plt.title("Entropy Increases During Mixing")
    plt.grid(True, alpha = 0.3)
    plt.tight_layout()
    if not no_save:
        plt.savefig(output_path, dpi = 150)
    if show:
        plt.show()
    plt.close()


def main():
    """Run the complexity dynamics demonstrations.

    Args:
        None: Uses command-line arguments from parse_args.

    Returns:
        None: Generates figures and prints summary values.
    """
    args = parse_args()
    set_seed(args.seed)

    output_dir = args.output_dir
    if not args.no_save:
        output_dir.mkdir(parents = True, exist_ok = True)

    logger.info("=" * 80)
    logger.info("Rule 30: Cellular Automaton")
    logger.info("=" * 80)

    initial = np.zeros(args.ca_size, dtype = int)
    initial[args.ca_size // 2] = 1
    evolution = evolve_ca(initial, args.ca_steps, rule_30)

    save_rule_30_figure(
        evolution,
        output_dir / "rule_30_evolution.png",
        show = args.show,
        no_save = args.no_save
    )

    logger.info("-" * 60)
    logger.info("Measuring Complexity Growth via Entropy")
    logger.info("-" * 60)

    entropies = measure_entropy_over_time(evolution)
    complexities = measure_spatial_complexity(evolution)
    save_entropy_complexity_figure(
        entropies,
        complexities,
        output_dir / "entropy_complexity.png",
        show = args.show,
        no_save = args.no_save
    )

    logger.info("*" * 50)
    logger.info("Coffee Automaton: Irreversible Mixing")
    logger.info("*" * 50)

    grid = np.zeros((args.diffusion_size, args.diffusion_size))
    start = args.diffusion_size // 2 - 5
    end = args.diffusion_size // 2 + 5
    grid[start:end, start:end] = 1.0
    mixing_history = diffusion_2d(
        grid,
        steps = args.diffusion_steps,
        diffusion_rate = args.diffusion_rate
    )

    save_coffee_grid_figure(
        mixing_history,
        output_dir / "coffee_mixing_grid.png",
        show = args.show,
        no_save = args.no_save
    )

    mixing_entropies = measure_mixing_entropy(mixing_history)
    save_mixing_entropy_figure(
        mixing_entropies,
        output_dir / "coffee_mixing_entropy.png",
        show = args.show,
        no_save = args.no_save
    )

    logger.info("Initial Entropy: %.4f bits", entropies[0])
    logger.info("Final Entropy: %.4f bits", entropies[-1])
    logger.info("Entropy Increase: %.4f bits", entropies[-1] - entropies[0])
    logger.info("Key Insight: Simple concentrated state -> Complex mixed state")
    logger.info("This process is irreversible: you can't unmix coffee!")


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers = [logging.StreamHandler()]
    )
    main()
