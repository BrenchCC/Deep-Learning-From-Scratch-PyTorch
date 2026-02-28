import os
import sys
import logging
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())

from chapter_06_rnn_lstm_seq.model import DynamicRNNClassifier
from chapter_06_rnn_lstm_seq.dataset import SyntheticNameDataset, VectorizedCollator

from utils import get_device
from utils import save_json
from utils import setup_seed
from utils import log_model_info
from utils import run_classification_epoch

logger = logging.getLogger("RNN_Experiment")


def parse_args():
    """
    Parse command-line arguments for chapter 06 training.

    Args:
        None

    Returns:
        argparse.Namespace: Parsed training configuration.
    """
    parser = argparse.ArgumentParser(description = "Train RNN/LSTM/GRU on Synthetic Data")
    parser.add_argument("--seed", type = int, default = 42, help = "Random seed")
    parser.add_argument("--epochs", type = int, default = 10, help = "Number of training epochs")
    parser.add_argument("--batch_size", type = int, default = 128, help = "Batch size")
    parser.add_argument("--hidden_dim", type = int, default = 256, help = "Hidden dimension size")
    parser.add_argument("--embed_dim", type = int, default = 128, help = "Embedding dimension size")
    parser.add_argument("--lr", type = float, default = 0.001, help = "Learning rate")
    parser.add_argument("--data_size", type = int, default = 10000, help = "Size of synthetic dataset")
    parser.add_argument("--val_split", type = float, default = 0.2, help = "Validation split ratio")
    parser.add_argument("--num_workers", type = int, default = 0, help = "DataLoader workers")
    parser.add_argument("--max_grad_norm", type = float, default = 1.0, help = "Gradient clipping max norm")
    parser.add_argument(
        "--model_type",
        type = str,
        default = "lstm",
        choices = ["rnn", "lstm", "gru"],
        help = "Model architecture"
    )
    return parser.parse_args()


def args_parser():
    """
    Backward-compatible alias for parse_args.

    Args:
        None

    Returns:
        argparse.Namespace: Parsed training configuration.
    """
    return parse_args()


def build_dataloaders(dataset, batch_size, val_split, num_workers):
    """
    Build train and validation DataLoaders from a single dataset.

    Args:
        dataset (SyntheticNameDataset): Full synthetic dataset.
        batch_size (int): Batch size.
        val_split (float): Validation ratio in (0, 1).
        num_workers (int): DataLoader worker count.

    Returns:
        tuple: (train_loader, val_loader)
    """
    if not 0.0 < val_split < 1.0:
        raise ValueError(f"val_split must be in (0, 1), got {val_split}")

    train_size = int((1.0 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    collator = VectorizedCollator(dataset.vocab)
    train_loader = DataLoader(
        train_set,
        batch_size = batch_size,
        shuffle = True,
        collate_fn = collator,
        num_workers = num_workers
    )
    val_loader = DataLoader(
        val_set,
        batch_size = batch_size,
        shuffle = False,
        collate_fn = collator,
        num_workers = num_workers
    )
    return train_loader, val_loader


def sequence_batch_adapter(batch, device):
    """
    Convert sequence batch to model arguments expected by DynamicRNNClassifier.

    Args:
        batch (tuple): (inputs, lengths, targets).
        device (torch.device): Runtime device.

    Returns:
        tuple: ((inputs, lengths_cpu), targets)
    """
    inputs, lengths, targets = batch
    inputs = inputs.to(device)
    lengths = lengths.to(torch.device("cpu"))
    targets = targets.to(device)
    return (inputs, lengths), targets


def evaluate_inference(model, vocab, classes, device, samples):
    """
    Run inference on a small set of names for quick sanity check.

    Args:
        model (DynamicRNNClassifier): Trained model.
        vocab (dict): Character to index mapping.
        classes (list): Class names.
        device (torch.device): Runtime device.
        samples (list): Input names.

    Returns:
        list: List of dict results.
    """
    model.eval()
    results = []

    logger.info("Running final inference sanity check...")
    with torch.no_grad():
        for name in samples:
            indices = [vocab.get(token, vocab["<unk>"]) for token in name]
            input_tensor = torch.tensor(indices, dtype = torch.long).unsqueeze(0).to(device)
            length_tensor = torch.tensor([len(indices)], dtype = torch.long)

            outputs = model(input_tensor, length_tensor)
            pred_idx = torch.argmax(outputs, dim = 1).item()
            pred_class = classes[pred_idx]
            logger.info(f"Input: {name:15s} | Prediction: {pred_class}")
            results.append({"input": name, "prediction": pred_class, "prediction_index": pred_idx})
    return results


def main():
    """
    Main training entry for chapter 06.

    Args:
        None

    Returns:
        None
    """
    args = parse_args()
    setup_seed(args.seed)
    device = get_device()

    base_dir = "chapter_06_rnn_lstm_seq"
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    res_dir = os.path.join(base_dir, "results")
    os.makedirs(ckpt_dir, exist_ok = True)
    os.makedirs(res_dir, exist_ok = True)

    logger.info("Initializing synthetic dataset...")
    dataset = SyntheticNameDataset(num_samples = args.data_size)
    dataset.save_vocab(os.path.join(base_dir, "data/vocab.txt"))
    dataset.save_data(os.path.join(base_dir, "data/synthetic_names.txt"))

    train_loader, val_loader = build_dataloaders(
        dataset = dataset,
        batch_size = args.batch_size,
        val_split = args.val_split,
        num_workers = args.num_workers
    )

    logger.info(f"Device: {device}")
    logger.info(f"Vocab Size: {len(dataset.vocab)} | Classes: {len(dataset.classes)}")
    logger.info(f"Hyperparameters: {vars(args)}")

    model = DynamicRNNClassifier(
        vocab_size = len(dataset.vocab),
        embedding_dim = args.embed_dim,
        hidden_dim = args.hidden_dim,
        output_dim = len(dataset.classes),
        num_layers = 2,
        bidirectional = True,
        dropout = 0.3,
        model_type = args.model_type
    ).to(device)
    log_model_info(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = args.lr)

    metrics = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_classification_epoch(
            model = model,
            dataloader = train_loader,
            criterion = criterion,
            device = device,
            stage = "train",
            optimizer = optimizer,
            epoch_idx = epoch,
            batch_adapter = sequence_batch_adapter,
            max_grad_norm = args.max_grad_norm
        )
        val_metrics = run_classification_epoch(
            model = model,
            dataloader = val_loader,
            criterion = criterion,
            device = device,
            stage = "eval",
            epoch_idx = epoch,
            batch_adapter = sequence_batch_adapter
        )

        metrics["train_loss"].append(train_metrics["loss"])
        metrics["train_acc"].append(train_metrics["acc"])
        metrics["val_loss"].append(val_metrics["loss"])
        metrics["val_acc"].append(val_metrics["acc"])

        logger.info(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['acc']:.2f}% | "
            f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['acc']:.2f}%"
        )

    model_path = os.path.join(ckpt_dir, f"{args.model_type}_model.pth")
    torch.save(model.state_dict(), model_path)
    metrics_path = os.path.join(res_dir, f"{args.model_type}_metrics.json")
    save_json(metrics, metrics_path)
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Metrics saved to {metrics_path}")

    test_samples = ["petrov", "jackson", "rossini", "yamamoto", "papadopoulos", "unknownstuff"]
    inference_results = evaluate_inference(model, dataset.vocab, dataset.classes, device, test_samples)
    save_json(inference_results, os.path.join(res_dir, f"{args.model_type}_inference_samples.json"))


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers = [logging.StreamHandler()]
    )
    main()
