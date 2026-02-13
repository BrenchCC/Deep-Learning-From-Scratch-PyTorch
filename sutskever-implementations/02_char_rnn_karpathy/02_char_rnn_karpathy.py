import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class VanillaRNN:
    def __init__(
        self,
        vocab_size,
        hidden_size,
        random_seed
    ):
        """
        Initialize the vanilla RNN parameters.

        Args:
            vocab_size: Number of unique characters.
            hidden_size: Number of hidden units.
            random_seed: Random seed for reproducibility.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        rng = np.random.default_rng(random_seed)

        self.Wxh = rng.standard_normal((hidden_size, vocab_size)) * 0.01
        self.Whh = rng.standard_normal((hidden_size, hidden_size)) * 0.01
        self.Why = rng.standard_normal((vocab_size, hidden_size)) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((vocab_size, 1))

    def forward(
        self,
        inputs,
        hprev
    ):
        """
        Run forward pass through a character sequence.

        Args:
            inputs: List of input character indices.
            hprev: Initial hidden state.
        """
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(hprev)

        for t, char_idx in enumerate(inputs):
            xs[t] = np.zeros((self.vocab_size, 1))
            xs[t][char_idx] = 1

            hs[t] = np.tanh(
                np.dot(self.Wxh, xs[t])
                + np.dot(self.Whh, hs[t - 1])
                + self.bh
            )

            ys[t] = np.dot(self.Why, hs[t]) + self.by
            exp_y = np.exp(ys[t] - np.max(ys[t]))
            ps[t] = exp_y / np.sum(exp_y)

        return xs, hs, ys, ps

    def loss(
        self,
        ps,
        targets
    ):
        """
        Compute cross-entropy loss.

        Args:
            ps: Predicted probability distributions per time step.
            targets: Target character indices.
        """
        total_loss = 0.0
        for t, target_idx in enumerate(targets):
            total_loss += -np.log(ps[t][target_idx, 0] + 1e-12)
        return total_loss

    def backward(
        self,
        xs,
        hs,
        ps,
        targets
    ):
        """
        Run backpropagation through time.

        Args:
            xs: Input one-hot vectors per time step.
            hs: Hidden states per time step.
            ps: Predicted probabilities per time step.
            targets: Target character indices.
        """
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        dhnext = np.zeros_like(hs[0])

        for t in reversed(range(len(targets))):
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1

            dWhy += np.dot(dy, hs[t].T)
            dby += dy

            dh = np.dot(self.Why.T, dy) + dhnext
            dhraw = (1 - hs[t] ** 2) * dh

            dbh += dhraw
            dWxh += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, hs[t - 1].T)

            dhnext = np.dot(self.Whh.T, dhraw)

        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out = dparam)

        return dWxh, dWhh, dWhy, dbh, dby

    def sample(
        self,
        h,
        seed_ix,
        n
    ):
        """
        Sample a sequence of character indices.

        Args:
            h: Initial hidden state.
            seed_ix: Seed character index.
            n: Number of characters to generate.
        """
        x = np.zeros((self.vocab_size, 1))
        x[seed_ix] = 1
        indices = []

        for _ in range(n):
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            y = np.dot(self.Why, h) + self.by
            exp_y = np.exp(y - np.max(y))
            p = exp_y / np.sum(exp_y)

            ix = np.random.choice(range(self.vocab_size), p = p.ravel())
            x = np.zeros((self.vocab_size, 1))
            x[ix] = 1
            indices.append(ix)

        return indices


def parse_args():
    """
    Parse command-line arguments.

    Args:
        None: No function parameters.
    """
    parser = argparse.ArgumentParser(description = "Character-level RNN from notebook.")
    parser.add_argument("--hidden-size", type = int, default = 64, help = "Hidden units.")
    parser.add_argument("--seq-length", type = int, default = 25, help = "Sequence length.")
    parser.add_argument("--num-iterations", type = int, default = 2000, help = "Training iterations.")
    parser.add_argument("--learning-rate", type = float, default = 0.1, help = "Learning rate.")
    parser.add_argument("--sample-every", type = int, default = 200, help = "Sample interval.")
    parser.add_argument("--sample-length", type = int, default = 100, help = "Sample text length.")
    parser.add_argument("--seed", type = int, default = 42, help = "Random seed.")
    parser.add_argument("--no-save", action = "store_true", help = "Disable writing files.")
    parser.add_argument("--show", action = "store_true", help = "Show figures interactively.")
    return parser.parse_args()


def build_training_data():
    """
    Build synthetic text data and vocabulary maps.

    Args:
        None: No function parameters.
    """
    data = (
        "\n"
        "hello world\n"
        "hello deep learning\n"
        "deep neural networks\n"
        "neural networks learn patterns\n"
        "patterns in data\n"
        "data drives learning\n"
        "learning from examples\n"
        "examples help networks\n"
        "networks process information\n"
        "information is everywhere\n"
        "everywhere you look data\n"
    ) * 10

    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}

    logger.info("Data length: %s characters", len(data))
    logger.info("Vocabulary size: %s", vocab_size)
    logger.info("Vocabulary: %s", repr("".join(chars)))

    return data, chars, vocab_size, char_to_ix, ix_to_char


def train_rnn(
    rnn,
    data,
    char_to_ix,
    ix_to_char,
    hidden_size,
    vocab_size,
    num_iterations,
    seq_length,
    learning_rate,
    sample_every,
    sample_length
):
    """
    Train the RNN with Adagrad.

    Args:
        rnn: The vanilla RNN model.
        data: Training text.
        char_to_ix: Character-to-index mapping.
        ix_to_char: Index-to-character mapping.
        hidden_size: Number of hidden units.
        vocab_size: Vocabulary size.
        num_iterations: Number of training iterations.
        seq_length: Sequence length per update.
        learning_rate: Learning rate for Adagrad.
        sample_every: Sampling interval for logging.
        sample_length: Characters generated per sample.
    """
    p = 0

    mWxh = np.zeros_like(rnn.Wxh)
    mWhh = np.zeros_like(rnn.Whh)
    mWhy = np.zeros_like(rnn.Why)
    mbh = np.zeros_like(rnn.bh)
    mby = np.zeros_like(rnn.by)

    smooth_loss = -np.log(1.0 / vocab_size) * seq_length
    losses = []
    sampled_texts = []
    hprev = np.zeros((hidden_size, 1))

    logger.info("*" * 50)
    logger.info("Model training started")
    logger.info("*" * 50)

    for n in tqdm(range(num_iterations), desc = "Training"):
        if p + seq_length + 1 >= len(data) or n == 0:
            hprev = np.zeros((hidden_size, 1))
            p = 0

        inputs = [char_to_ix[ch] for ch in data[p : p + seq_length]]
        targets = [char_to_ix[ch] for ch in data[p + 1 : p + seq_length + 1]]

        xs, hs, _, ps = rnn.forward(inputs, hprev)
        loss = rnn.loss(ps, targets)
        dWxh, dWhh, dWhy, dbh, dby = rnn.backward(xs, hs, ps, targets)

        for param, dparam, mem in zip(
            [rnn.Wxh, rnn.Whh, rnn.Why, rnn.bh, rnn.by],
            [dWxh, dWhh, dWhy, dbh, dby],
            [mWxh, mWhh, mWhy, mbh, mby]
        ):
            mem += dparam * dparam
            param += -learning_rate * dparam / np.sqrt(mem + 1e-8)

        smooth_loss = smooth_loss * 0.999 + loss * 0.001
        losses.append(float(smooth_loss))

        if n % sample_every == 0:
            sample_ix = rnn.sample(hprev, inputs[0], sample_length)
            txt = "".join(ix_to_char[ix] for ix in sample_ix)
            sampled_texts.append(
                {
                    "iteration": int(n),
                    "loss": float(smooth_loss),
                    "sample": txt
                }
            )
            logger.info("-" * 60)
            logger.info("Iteration %s, Loss %.4f", n, smooth_loss)
            logger.info("%s", txt)

        p += seq_length
        hprev = hs[len(inputs) - 1]

    return losses, sampled_texts


def plot_training_loss(
    losses,
    image_path,
    show
):
    """
    Plot and optionally save the loss curve.

    Args:
        losses: Smooth loss values.
        image_path: Output image path.
        show: Whether to display figure.
    """
    plt.figure(figsize = (12, 5))
    plt.plot(losses, linewidth = 2)
    plt.xlabel("Iteration")
    plt.ylabel("Smooth Loss")
    plt.title("RNN Training Loss (Character-Level Language Model)")
    plt.grid(True, alpha = 0.3)
    plt.tight_layout()
    plt.savefig(image_path, dpi = 150)

    if show:
        plt.show()

    plt.close()


def generate_samples(
    rnn,
    chars,
    char_to_ix,
    ix_to_char,
    hidden_size,
    num_samples,
    sample_length
):
    """
    Generate text samples from trained model.

    Args:
        rnn: The trained vanilla RNN model.
        chars: Vocabulary characters.
        char_to_ix: Character-to-index mapping.
        ix_to_char: Index-to-character mapping.
        hidden_size: Number of hidden units.
        num_samples: Number of generated samples.
        sample_length: Number of characters per sample.
    """
    h = np.zeros((hidden_size, 1))
    samples = []

    logger.info("Generated samples:")
    for i in range(num_samples):
        seed_char = np.random.choice(chars)
        seed_ix = char_to_ix[seed_char]
        sample_ix = rnn.sample(h, seed_ix, sample_length)
        txt = "".join(ix_to_char[ix] for ix in sample_ix)

        samples.append(
            {
                "sample_id": i + 1,
                "seed_char": seed_char,
                "text": txt
            }
        )
        logger.info("Sample %s (seed: '%s')", i + 1, seed_char)
        logger.info("%s", txt)

    return samples


def plot_hidden_states(
    rnn,
    test_text,
    char_to_ix,
    hidden_size,
    image_path,
    show
):
    """
    Visualize hidden state activations for a test sequence.

    Args:
        rnn: The trained vanilla RNN model.
        test_text: Input text for activation analysis.
        char_to_ix: Character-to-index mapping.
        hidden_size: Number of hidden units.
        image_path: Output image path.
        show: Whether to display figure.
    """
    test_inputs = [char_to_ix[ch] for ch in test_text]
    hprev = np.zeros((hidden_size, 1))

    _, hs, _, _ = rnn.forward(test_inputs, hprev)
    hidden_states = np.array([hs[t].flatten() for t in range(len(test_inputs))])

    plt.figure(figsize = (14, 6))
    plt.imshow(
        hidden_states.T,
        cmap = "RdBu",
        aspect = "auto",
        interpolation = "nearest"
    )
    plt.colorbar(label = "Activation")
    plt.xlabel("Time Step (Character Position)")
    plt.ylabel("Hidden Unit")
    plt.title("RNN Hidden State Activations")
    plt.xticks(range(len(test_text)), list(test_text))
    plt.tight_layout()
    plt.savefig(image_path, dpi = 150)

    if show:
        plt.show()

    plt.close()
    logger.info(
        "Visualization shows how hidden states evolve as RNN processes '%s'",
        test_text
    )


def ensure_output_dirs(base_dir):
    """
    Create required output directories.

    Args:
        base_dir: Base directory of the script.
    """
    images_dir = base_dir / "images"
    results_dir = base_dir / "results"
    models_dir = base_dir / "checkpoints" / "models"

    images_dir.mkdir(parents = True, exist_ok = True)
    results_dir.mkdir(parents = True, exist_ok = True)
    models_dir.mkdir(parents = True, exist_ok = True)

    return images_dir, results_dir, models_dir


def save_results(
    results_dir,
    losses,
    sampled_texts,
    generated_samples
):
    """
    Save training and generation outputs to JSON files.

    Args:
        results_dir: Directory for result files.
        losses: Smooth loss values.
        sampled_texts: Periodic training samples.
        generated_samples: Final generated samples.
    """
    summary = {
        "final_loss": float(losses[-1]),
        "min_loss": float(min(losses)),
        "num_steps": len(losses)
    }

    (results_dir / "training_losses.json").write_text(
        json.dumps(losses, indent = 2),
        encoding = "utf-8"
    )
    (results_dir / "training_samples.json").write_text(
        json.dumps(sampled_texts, indent = 2, ensure_ascii = False),
        encoding = "utf-8"
    )
    (results_dir / "generated_samples.json").write_text(
        json.dumps(generated_samples, indent = 2, ensure_ascii = False),
        encoding = "utf-8"
    )
    (results_dir / "summary.json").write_text(
        json.dumps(summary, indent = 2),
        encoding = "utf-8"
    )


def main(args):
    """
    Execute full notebook-to-script workflow.

    Args:
        args: Parsed command-line arguments.
    """
    np.random.seed(args.seed)

    script_dir = Path(__file__).resolve().parent
    images_dir, results_dir, models_dir = ensure_output_dirs(script_dir)

    logger.info("=" * 80)
    logger.info("Character-Level Language Model with Vanilla RNN")
    logger.info("=" * 80)

    data, chars, vocab_size, char_to_ix, ix_to_char = build_training_data()

    logger.info("\nModel initialized with %s hidden units", args.hidden_size)
    rnn = VanillaRNN(
        vocab_size = vocab_size,
        hidden_size = args.hidden_size,
        random_seed = args.seed
    )

    losses, sampled_texts = train_rnn(
        rnn = rnn,
        data = data,
        char_to_ix = char_to_ix,
        ix_to_char = ix_to_char,
        hidden_size = args.hidden_size,
        vocab_size = vocab_size,
        num_iterations = args.num_iterations,
        seq_length = args.seq_length,
        learning_rate = args.learning_rate,
        sample_every = args.sample_every,
        sample_length = args.sample_length
    )

    plot_training_loss(
        losses = losses,
        image_path = images_dir / "training_loss.png",
        show = args.show
    )

    generated_samples = generate_samples(
        rnn = rnn,
        chars = chars,
        char_to_ix = char_to_ix,
        ix_to_char = ix_to_char,
        hidden_size = args.hidden_size,
        num_samples = 5,
        sample_length = 150
    )

    plot_hidden_states(
        rnn = rnn,
        test_text = "hello deep learning",
        char_to_ix = char_to_ix,
        hidden_size = args.hidden_size,
        image_path = images_dir / "hidden_state_activations.png",
        show = args.show
    )

    if not args.no_save:
        save_results(
            results_dir = results_dir,
            losses = losses,
            sampled_texts = sampled_texts,
            generated_samples = generated_samples
        )

        model_payload = {
            "Wxh": rnn.Wxh.tolist(),
            "Whh": rnn.Whh.tolist(),
            "Why": rnn.Why.tolist(),
            "bh": rnn.bh.tolist(),
            "by": rnn.by.tolist()
        }
        (models_dir / "vanilla_rnn_weights.json").write_text(
            json.dumps(model_payload),
            encoding = "utf-8"
        )

    logger.info("Outputs saved under: %s", script_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers = [logging.StreamHandler()]
    )
    cli_args = parse_args()
    main(cli_args)
