import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

logger = logging.getLogger(__name__)


class LSTMCell:
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        weight_scale: float = 0.01
    ) -> None:
        """Initialize one LSTM cell.

        Args:
            input_size: Size of each input vector.
            hidden_size: Size of hidden state and cell state.
            weight_scale: Scale factor for random weight initialization.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size

        concat_size = input_size + hidden_size

        self.Wf = np.random.randn(hidden_size, concat_size) * weight_scale
        self.bf = np.zeros((hidden_size, 1))

        self.Wi = np.random.randn(hidden_size, concat_size) * weight_scale
        self.bi = np.zeros((hidden_size, 1))

        self.Wc = np.random.randn(hidden_size, concat_size) * weight_scale
        self.bc = np.zeros((hidden_size, 1))

        self.Wo = np.random.randn(hidden_size, concat_size) * weight_scale
        self.bo = np.zeros((hidden_size, 1))

    def forward(
        self,
        x: np.ndarray,
        h_prev: np.ndarray,
        c_prev: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """Run one forward step for the LSTM cell.

        Args:
            x: Current input vector with shape (input_size, 1).
            h_prev: Previous hidden state with shape (hidden_size, 1).
            c_prev: Previous cell state with shape (hidden_size, 1).

        Returns:
            Next hidden state, next cell state, and cached gate/state values.
        """
        concat = np.vstack([x, h_prev])

        f = sigmoid(np.dot(self.Wf, concat) + self.bf)
        i = sigmoid(np.dot(self.Wi, concat) + self.bi)
        c_tilde = np.tanh(np.dot(self.Wc, concat) + self.bc)

        c_next = f * c_prev + i * c_tilde

        o = sigmoid(np.dot(self.Wo, concat) + self.bo)
        h_next = o * np.tanh(c_next)

        cache = {
            "x": x,
            "h_prev": h_prev,
            "c_prev": c_prev,
            "concat": concat,
            "f": f,
            "i": i,
            "c_tilde": c_tilde,
            "c_next": c_next,
            "o": o,
            "h_next": h_next,
        }

        return h_next, c_next, cache


class LSTM:
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        weight_scale: float = 0.01
    ) -> None:
        """Initialize a simple LSTM network for sequence processing.

        Args:
            input_size: Size of each input vector.
            hidden_size: Size of hidden state and cell state.
            output_size: Size of final output vector.
            weight_scale: Scale factor for random weight initialization.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.cell = LSTMCell(
            input_size = input_size,
            hidden_size = hidden_size,
            weight_scale = weight_scale
        )

        self.Why = np.random.randn(output_size, hidden_size) * weight_scale
        self.by = np.zeros((output_size, 1))

    def forward(
        self,
        inputs: List[np.ndarray],
        use_tqdm: bool = False,
        desc: str = "LSTM forward"
    ) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], Dict[str, List[np.ndarray]]]:
        """Process a whole sequence through the LSTM.

        Args:
            inputs: Sequence of input vectors.
            use_tqdm: Whether to display progress bars for sequence steps.
            desc: Progress bar description.

        Returns:
            Final output, hidden states, cell states, and gate values by time step.
        """
        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))

        h_states: List[np.ndarray] = []
        c_states: List[np.ndarray] = []
        gate_values = {
            "f": [],
            "i": [],
            "o": [],
        }

        iterator = tqdm(inputs, desc = desc, leave = False) if use_tqdm else inputs
        for x in iterator:
            h, c, cache = self.cell.forward(x = x, h_prev = h, c_prev = c)
            h_states.append(h.copy())
            c_states.append(c.copy())
            gate_values["f"].append(cache["f"].copy())
            gate_values["i"].append(cache["i"].copy())
            gate_values["o"].append(cache["o"].copy())

        y = np.dot(self.Why, h) + self.by
        return y, h_states, c_states, gate_values


class VanillaRNNCell:
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        weight_scale: float = 0.01
    ) -> None:
        """Initialize a vanilla RNN cell for baseline comparison.

        Args:
            input_size: Size of each input vector.
            hidden_size: Size of hidden state.
            weight_scale: Scale factor for random weight initialization.
        """
        concat_size = input_size + hidden_size
        self.hidden_size = hidden_size
        self.Wh = np.random.randn(hidden_size, concat_size) * weight_scale
        self.bh = np.zeros((hidden_size, 1))

    def forward(self, x: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        """Run one forward step for the vanilla RNN cell.

        Args:
            x: Current input vector with shape (input_size, 1).
            h_prev: Previous hidden state with shape (hidden_size, 1).

        Returns:
            Next hidden state.
        """
        concat = np.vstack([x, h_prev])
        h_next = np.tanh(np.dot(self.Wh, concat) + self.bh)
        return h_next


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the LSTM experiment.

    Args:
        None: This function reads values from command line only.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description = "LSTM gate visualization and long-term dependency demonstration"
    )
    parser.add_argument("--seed", type = int, default = 42)
    parser.add_argument("--input-size", type = int, default = 5)
    parser.add_argument("--hidden-size", type = int, default = 16)
    parser.add_argument("--output-size", type = int, default = 5)
    parser.add_argument("--seq-length", type = int, default = 15)
    parser.add_argument("--num-samples", type = int, default = 10)
    parser.add_argument("--gradient-steps", type = int, default = 30)
    parser.add_argument("--rnn-decay", type = float, default = 0.85)
    parser.add_argument("--lstm-forget", type = float, default = 0.95)
    parser.add_argument(
        "--output-dir",
        type = str,
        default = str(Path(__file__).resolve().parent)
    )
    parser.add_argument("--use-tqdm", action = "store_true")
    parser.add_argument("--no-save", action = "store_true")
    parser.add_argument("--show", action = "store_true")
    return parser.parse_args()


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Compute the sigmoid activation.

    Args:
        x: Input array.

    Returns:
        Sigmoid output array.
    """
    return 1.0 / (1.0 + np.exp(-x))


def ensure_output_dirs(output_dir: Path) -> Dict[str, Path]:
    """Create and return required output directories.

    Args:
        output_dir: Base directory where all outputs are stored.

    Returns:
        Dictionary containing image, result, and model directories.
    """
    images_dir = output_dir / "images"
    results_dir = output_dir / "results"
    models_dir = output_dir / "checkpoints" / "models"

    images_dir.mkdir(parents = True, exist_ok = True)
    results_dir.mkdir(parents = True, exist_ok = True)
    models_dir.mkdir(parents = True, exist_ok = True)

    return {
        "images": images_dir,
        "results": results_dir,
        "models": models_dir,
    }


def generate_long_term_dependency_data(
    seq_length: int,
    num_samples: int,
    input_size: int,
    output_size: int,
    use_tqdm: bool = False
) -> Tuple[List[List[np.ndarray]], List[np.ndarray]]:
    """Generate synthetic data for long-term dependency testing.

    Args:
        seq_length: Number of time steps per sequence.
        num_samples: Number of sequences to generate.
        input_size: Feature dimension for each input vector.
        output_size: Number of classes for target one-hot vector.
        use_tqdm: Whether to display progress while generating samples.

    Returns:
        List of sequences and corresponding one-hot targets.
    """
    X: List[List[np.ndarray]] = []
    y: List[np.ndarray] = []

    sample_range = range(num_samples)
    iterator = tqdm(sample_range, desc = "Generate samples", leave = False) if use_tqdm else sample_range

    for _ in iterator:
        sequence: List[np.ndarray] = []

        first_elem = np.random.randint(0, input_size)
        first_vec = np.zeros((input_size, 1))
        first_vec[first_elem] = 1.0
        sequence.append(first_vec)

        for _ in range(seq_length - 1):
            noise = np.random.randn(input_size, 1) * 0.1
            sequence.append(noise)

        target = np.zeros((output_size, 1))
        target[first_elem] = 1.0

        X.append(sequence)
        y.append(target)

    return X, y


def process_with_vanilla_rnn(
    rnn_cell: VanillaRNNCell,
    inputs: List[np.ndarray],
    hidden_size: int,
    use_tqdm: bool = False
) -> List[np.ndarray]:
    """Run a sequence through a vanilla RNN cell.

    Args:
        rnn_cell: Vanilla RNN cell instance.
        inputs: Sequence of input vectors.
        hidden_size: Hidden state dimension.
        use_tqdm: Whether to display progress for sequence processing.

    Returns:
        Hidden state list across time steps.
    """
    h = np.zeros((hidden_size, 1))
    h_states: List[np.ndarray] = []

    iterator = tqdm(inputs, desc = "RNN forward", leave = False) if use_tqdm else inputs
    for x in iterator:
        h = rnn_cell.forward(x = x, h_prev = h)
        h_states.append(h.copy())

    return h_states


def simulate_gradient_flow(
    seq_length: int,
    rnn_decay: float,
    lstm_forget: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate gradient magnitude decay for RNN and LSTM.

    Args:
        seq_length: Number of backpropagation steps.
        rnn_decay: Multiplicative gradient decay factor for vanilla RNN.
        lstm_forget: Multiplicative factor approximating LSTM gradient retention.

    Returns:
        Simulated gradient arrays for vanilla RNN and LSTM.
    """
    rnn_grads: List[float] = []
    grad = 1.0
    for _ in range(seq_length):
        rnn_grads.append(grad)
        grad *= rnn_decay

    lstm_grads: List[float] = []
    grad = 1.0
    for _ in range(seq_length):
        lstm_grads.append(grad)
        grad *= lstm_forget

    return np.array(rnn_grads), np.array(lstm_grads)


def visualize_lstm_gates(
    gate_values: Dict[str, List[np.ndarray]],
    c_states: List[np.ndarray],
    h_states: List[np.ndarray],
    save_path: Path,
    save_figure: bool,
    show_figure: bool
) -> Dict[str, float]:
    """Visualize LSTM gate and state dynamics across time.

    Args:
        gate_values: Dictionary containing forget/input/output gate histories.
        c_states: Cell states over time.
        h_states: Hidden states over time.
        save_path: Output path for the image file.
        save_figure: Whether to save the figure to disk.
        show_figure: Whether to show the figure interactively.

    Returns:
        Basic summary statistics for gate activations.
    """
    forget_gates = np.hstack(gate_values["f"])
    input_gates = np.hstack(gate_values["i"])
    output_gates = np.hstack(gate_values["o"])
    cell_states = np.hstack(c_states)
    hidden_states = np.hstack(h_states)

    fig, axes = plt.subplots(5, 1, figsize = (14, 12))

    axes[0].imshow(forget_gates, cmap = "RdYlGn", aspect = "auto", vmin = 0, vmax = 1)
    axes[0].set_title("Forget Gate (1 = keep, 0 = forget)")
    axes[0].set_ylabel("Hidden Unit")
    axes[0].set_xlabel("Time Step")

    axes[1].imshow(input_gates, cmap = "RdYlGn", aspect = "auto", vmin = 0, vmax = 1)
    axes[1].set_title("Input Gate (1 = accept new, 0 = ignore)")
    axes[1].set_ylabel("Hidden Unit")
    axes[1].set_xlabel("Time Step")

    axes[2].imshow(output_gates, cmap = "RdYlGn", aspect = "auto", vmin = 0, vmax = 1)
    axes[2].set_title("Output Gate (1 = expose, 0 = hide)")
    axes[2].set_ylabel("Hidden Unit")
    axes[2].set_xlabel("Time Step")

    im3 = axes[3].imshow(cell_states, cmap = "RdBu", aspect = "auto")
    axes[3].set_title("Cell State (Long-term Memory)")
    axes[3].set_ylabel("Hidden Unit")
    axes[3].set_xlabel("Time Step")
    plt.colorbar(im3, ax = axes[3])

    im4 = axes[4].imshow(hidden_states, cmap = "RdBu", aspect = "auto")
    axes[4].set_title("Hidden State (Output to Next Layer)")
    axes[4].set_ylabel("Hidden Unit")
    axes[4].set_xlabel("Time Step")
    plt.colorbar(im4, ax = axes[4])

    plt.tight_layout()
    if save_figure:
        fig.savefig(save_path, dpi = 200, bbox_inches = "tight")
    if show_figure:
        plt.show()
    plt.close(fig)

    return {
        "forget_gate_mean": float(np.mean(forget_gates)),
        "input_gate_mean": float(np.mean(input_gates)),
        "output_gate_mean": float(np.mean(output_gates)),
        "cell_state_std": float(np.std(cell_states)),
        "hidden_state_std": float(np.std(hidden_states)),
    }


def visualize_lstm_vs_rnn(
    rnn_hidden: np.ndarray,
    lstm_hidden: np.ndarray,
    save_path: Path,
    save_figure: bool,
    show_figure: bool
) -> Dict[str, float]:
    """Visualize hidden state dynamics for vanilla RNN and LSTM.

    Args:
        rnn_hidden: Hidden state matrix from vanilla RNN.
        lstm_hidden: Hidden state matrix from LSTM.
        save_path: Output path for the image file.
        save_figure: Whether to save the figure to disk.
        show_figure: Whether to show the figure interactively.

    Returns:
        Summary metrics for hidden state variability.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 5))

    im1 = ax1.imshow(rnn_hidden, cmap = "RdBu", aspect = "auto")
    ax1.set_title("Vanilla RNN Hidden States")
    ax1.set_ylabel("Hidden Unit")
    ax1.set_xlabel("Time Step")
    plt.colorbar(im1, ax = ax1)

    im2 = ax2.imshow(lstm_hidden, cmap = "RdBu", aspect = "auto")
    ax2.set_title("LSTM Hidden States")
    ax2.set_ylabel("Hidden Unit")
    ax2.set_xlabel("Time Step")
    plt.colorbar(im2, ax = ax2)

    plt.tight_layout()
    if save_figure:
        fig.savefig(save_path, dpi = 200, bbox_inches = "tight")
    if show_figure:
        plt.show()
    plt.close(fig)

    return {
        "rnn_hidden_std": float(np.std(rnn_hidden)),
        "lstm_hidden_std": float(np.std(lstm_hidden)),
    }


def visualize_gradient_flow(
    rnn_grads: np.ndarray,
    lstm_grads: np.ndarray,
    save_path: Path,
    save_figure: bool,
    show_figure: bool
) -> Dict[str, float]:
    """Visualize simulated gradient flow for vanilla RNN and LSTM.

    Args:
        rnn_grads: Gradient magnitudes for vanilla RNN.
        lstm_grads: Gradient magnitudes for LSTM.
        save_path: Output path for the image file.
        save_figure: Whether to save the figure to disk.
        show_figure: Whether to show the figure interactively.

    Returns:
        Final-step gradient magnitudes for both models.
    """
    fig = plt.figure(figsize = (12, 5))
    plt.plot(rnn_grads[::-1], label = "Vanilla RNN", linewidth = 2)
    plt.plot(lstm_grads[::-1], label = "LSTM", linewidth = 2)
    plt.xlabel("Timesteps in the Past")
    plt.ylabel("Gradient Magnitude")
    plt.title("Gradient Flow: LSTM vs Vanilla RNN")
    plt.legend()
    plt.grid(True, alpha = 0.3)
    plt.yscale("log")

    plt.tight_layout()
    if save_figure:
        fig.savefig(save_path, dpi = 200, bbox_inches = "tight")
    if show_figure:
        plt.show()
    plt.close(fig)

    return {
        "rnn_final_gradient": float(rnn_grads[-1]),
        "lstm_final_gradient": float(lstm_grads[-1]),
    }


def save_model_artifacts(
    lstm_model: LSTM,
    rnn_cell: VanillaRNNCell,
    model_dir: Path
) -> List[str]:
    """Save model parameter artifacts for reproducibility.

    Args:
        lstm_model: LSTM model instance.
        rnn_cell: Vanilla RNN cell instance.
        model_dir: Directory used for model artifacts.

    Returns:
        List of generated model artifact paths.
    """
    artifact_paths: List[str] = []

    lstm_weights_path = model_dir / "lstm_init_weights.npz"
    np.savez(
        lstm_weights_path,
        Wf = lstm_model.cell.Wf,
        bf = lstm_model.cell.bf,
        Wi = lstm_model.cell.Wi,
        bi = lstm_model.cell.bi,
        Wc = lstm_model.cell.Wc,
        bc = lstm_model.cell.bc,
        Wo = lstm_model.cell.Wo,
        bo = lstm_model.cell.bo,
        Why = lstm_model.Why,
        by = lstm_model.by,
    )
    artifact_paths.append(str(lstm_weights_path))

    rnn_weights_path = model_dir / "vanilla_rnn_init_weights.npz"
    np.savez(rnn_weights_path, Wh = rnn_cell.Wh, bh = rnn_cell.bh)
    artifact_paths.append(str(rnn_weights_path))

    return artifact_paths


def save_results(summary: Dict[str, object], results_dir: Path) -> str:
    """Save experiment summary to a JSON file.

    Args:
        summary: Experiment summary dictionary.
        results_dir: Directory used for result files.

    Returns:
        Path to the written JSON summary file.
    """
    summary_path = results_dir / "lstm_understanding_summary.json"
    with summary_path.open("w", encoding = "utf-8") as fp:
        json.dump(summary, fp, indent = 2, ensure_ascii = False)
    return str(summary_path)


def run_experiment(args: argparse.Namespace) -> Dict[str, object]:
    """Run the full LSTM understanding experiment pipeline.

    Args:
        args: Parsed command line arguments.

    Returns:
        A dictionary summarizing generated outputs and key metrics.
    """
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir).resolve()
    dirs = ensure_output_dirs(output_dir = output_dir)
    save_figure = not args.no_save

    logger.info("=" * 80)
    logger.info("Section 1 - Initialize LSTM Cell")
    logger.info("=" * 80)

    lstm_cell = LSTMCell(
        input_size = args.input_size,
        hidden_size = args.hidden_size
    )
    x = np.random.randn(args.input_size, 1)
    h = np.zeros((args.hidden_size, 1))
    c = np.zeros((args.hidden_size, 1))
    h_next, c_next, _ = lstm_cell.forward(x = x, h_prev = h, c_prev = c)

    logger.info("LSTM Cell initialized: input_size = %d, hidden_size = %d", args.input_size, args.hidden_size)
    logger.info("Hidden state shape: %s", h_next.shape)
    logger.info("Cell state shape: %s", c_next.shape)

    logger.info("-" * 60)
    logger.info("Section 2 - Build Full LSTM Model")
    logger.info("-" * 60)

    lstm_model = LSTM(
        input_size = args.input_size,
        hidden_size = args.hidden_size,
        output_size = args.output_size
    )
    logger.info(
        "LSTM model created: %d -> %d -> %d",
        args.input_size,
        args.hidden_size,
        args.output_size
    )

    logger.info("-" * 60)
    logger.info("Section 3 - Long-Term Dependency Data")
    logger.info("-" * 60)

    X_test, y_test = generate_long_term_dependency_data(
        seq_length = args.seq_length,
        num_samples = args.num_samples,
        input_size = args.input_size,
        output_size = args.output_size,
        use_tqdm = args.use_tqdm
    )

    output, h_states, c_states, gate_values = lstm_model.forward(
        inputs = X_test[0],
        use_tqdm = args.use_tqdm,
        desc = "LSTM test forward"
    )

    logger.info("Test sequence length: %d", len(X_test[0]))
    logger.info("First element (to remember): %d", int(np.argmax(X_test[0][0])))
    logger.info("Expected output index: %d", int(np.argmax(y_test[0])))
    logger.info("Model output (untrained, first 5 dims): %s", output.flatten()[:5])

    logger.info("*" * 50)
    logger.info("Section 4 - Visualize LSTM Gates")
    logger.info("*" * 50)

    gate_figure_path = dirs["images"] / "lstm_gate_visualization.png"
    gate_stats = visualize_lstm_gates(
        gate_values = gate_values,
        c_states = c_states,
        h_states = h_states,
        save_path = gate_figure_path,
        save_figure = save_figure,
        show_figure = args.show
    )

    logger.info("Gate interpretation:")
    logger.info("Forget gate controls what information is discarded.")
    logger.info("Input gate controls what new information enters memory.")
    logger.info("Output gate controls what memory content is exposed.")
    logger.info("Cell state acts as long-term memory highway.")

    logger.info("*" * 50)
    logger.info("Section 5 - Compare LSTM vs Vanilla RNN")
    logger.info("*" * 50)

    rnn_cell = VanillaRNNCell(
        input_size = args.input_size,
        hidden_size = args.hidden_size
    )
    rnn_h_states = process_with_vanilla_rnn(
        rnn_cell = rnn_cell,
        inputs = X_test[0],
        hidden_size = args.hidden_size,
        use_tqdm = args.use_tqdm
    )

    rnn_hidden = np.hstack(rnn_h_states)
    lstm_hidden = np.hstack(h_states)

    compare_figure_path = dirs["images"] / "lstm_vs_vanilla_rnn_states.png"
    compare_stats = visualize_lstm_vs_rnn(
        rnn_hidden = rnn_hidden,
        lstm_hidden = lstm_hidden,
        save_path = compare_figure_path,
        save_figure = save_figure,
        show_figure = args.show
    )

    logger.info("Key differences:")
    logger.info("LSTM keeps separate cell state and hidden state.")
    logger.info("Gates provide selective information routing.")
    logger.info("Gradient flow is better preserved across time.")

    logger.info("*" * 50)
    logger.info("Section 6 - Gradient Flow Comparison")
    logger.info("*" * 50)

    rnn_grads, lstm_grads = simulate_gradient_flow(
        seq_length = args.gradient_steps,
        rnn_decay = args.rnn_decay,
        lstm_forget = args.lstm_forget
    )

    grad_figure_path = dirs["images"] / "gradient_flow_comparison.png"
    grad_stats = visualize_gradient_flow(
        rnn_grads = rnn_grads,
        lstm_grads = lstm_grads,
        save_path = grad_figure_path,
        save_figure = save_figure,
        show_figure = args.show
    )

    logger.info("Gradient after %d steps:", args.gradient_steps)
    logger.info("Vanilla RNN: %.6f (vanishing)", grad_stats["rnn_final_gradient"])
    logger.info("LSTM: %.6f (preserved)", grad_stats["lstm_final_gradient"])

    model_artifacts = []
    if save_figure:
        model_artifacts = save_model_artifacts(
            lstm_model = lstm_model,
            rnn_cell = rnn_cell,
            model_dir = dirs["models"]
        )

    summary = {
        "seed": args.seed,
        "input_size": args.input_size,
        "hidden_size": args.hidden_size,
        "output_size": args.output_size,
        "seq_length": args.seq_length,
        "num_samples": args.num_samples,
        "gradient_steps": args.gradient_steps,
        "save_enabled": save_figure,
        "show_enabled": args.show,
        "figures": [
            str(gate_figure_path),
            str(compare_figure_path),
            str(grad_figure_path),
        ] if save_figure else [],
        "gate_stats": gate_stats,
        "comparison_stats": compare_stats,
        "gradient_stats": grad_stats,
        "model_artifacts": model_artifacts,
    }

    summary_path = save_results(summary = summary, results_dir = dirs["results"])
    summary["summary_path"] = summary_path
    return summary


def main() -> None:
    """Entry point for the script execution.

    Args:
        None: This function reads parsed CLI args and launches the experiment.

    Returns:
        None.
    """
    args = parse_args()

    logging.basicConfig(
        level = logging.INFO,
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers = [logging.StreamHandler()]
    )

    summary = run_experiment(args = args)

    logger.info("=" * 80)
    logger.info("Run completed. Summary file: %s", summary["summary_path"])
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
