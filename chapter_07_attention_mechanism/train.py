import os
import sys
import json
import logging
import argparse
from typing import Dict
from typing import List
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
from tqdm import tqdm

# Add project root to Python path
sys.path.append(os.getcwd())

from chapter_07_attention_mechanism.dataset import PAD_TOKEN_ID
from chapter_07_attention_mechanism.dataset import MASK_TOKEN_ID
from chapter_07_attention_mechanism.dataset import MaskedCopyDataset
from chapter_07_attention_mechanism.dataset import MaskedCopyCollator
from chapter_07_attention_mechanism.model import SingleLayerSelfAttentionModel
from utils import get_device
from utils import setup_seed

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Args:
        None

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description = "Chapter 07 masked copy training")
    parser.add_argument("--seed", type = int, default = 42, help = "Random seed")
    parser.add_argument("--epochs", type = int, default = 8, help = "Training epochs")
    parser.add_argument("--batch_size", type = int, default = 64, help = "Batch size")
    parser.add_argument("--num_samples", type = int, default = 10000, help = "Dataset sample size")
    parser.add_argument("--val_split", type = float, default = 0.2, help = "Validation split")
    parser.add_argument("--vocab_size", type = int, default = 64, help = "Vocabulary size")
    parser.add_argument("--min_seq_len", type = int, default = 6, help = "Min sequence length")
    parser.add_argument("--max_seq_len", type = int, default = 20, help = "Max sequence length")
    parser.add_argument("--mask_ratio", type = float, default = 0.3, help = "Masked ratio")
    parser.add_argument("--d_model", type = int, default = 128, help = "Model hidden size")
    parser.add_argument("--dropout", type = float, default = 0.1, help = "Dropout")
    parser.add_argument("--lr", type = float, default = 1e-3, help = "Learning rate")
    parser.add_argument("--weight_decay", type = float, default = 1e-4, help = "Weight decay")
    parser.add_argument("--max_grad_norm", type = float, default = 1.0, help = "Grad clip")
    parser.add_argument("--num_workers", type = int, default = 0, help = "DataLoader workers")
    parser.add_argument(
        "--num_prediction_examples",
        type = int,
        default = 5,
        help = "Number of prediction examples"
    )
    parser.add_argument(
        "--result_dir",
        type = str,
        default = "chapter_07_attention_mechanism/results",
        help = "Directory for result files"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type = str,
        default = "chapter_07_attention_mechanism/checkpoints",
        help = "Directory for checkpoints"
    )
    return parser.parse_args()


def compute_masked_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    predict_mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute cross-entropy only on masked positions.

    Args:
        logits (torch.Tensor): Model logits [batch, seq_len, vocab_size].
        targets (torch.Tensor): Target ids [batch, seq_len].
        predict_mask (torch.Tensor): Masked positions [batch, seq_len].

    Returns:
        torch.Tensor: Scalar loss.
    """
    vocab_size = logits.size(-1)
    per_token_loss = nn.functional.cross_entropy(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1),
        reduction = "none"
    )

    flat_mask = predict_mask.reshape(-1)
    valid_count = flat_mask.sum().clamp_min(1)
    masked_loss = (per_token_loss * flat_mask.float()).sum() / valid_count.float()
    return masked_loss


def compute_masked_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    predict_mask: torch.Tensor
) -> Tuple[int, int]:
    """
    Compute accuracy on masked positions.

    Args:
        logits (torch.Tensor): Model logits [batch, seq_len, vocab_size].
        targets (torch.Tensor): Target ids [batch, seq_len].
        predict_mask (torch.Tensor): Masked positions [batch, seq_len].

    Returns:
        Tuple[int, int]: Correct count and total count.
    """
    predictions = logits.argmax(dim = -1)
    correct = predictions.eq(targets) & predict_mask
    correct_count = int(correct.sum().item())
    total_count = int(predict_mask.sum().item())
    return correct_count, total_count


def run_epoch(
    model: SingleLayerSelfAttentionModel,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    stage: str,
    max_grad_norm: float
) -> Dict[str, float]:
    """
    Run one training or validation epoch.

    Args:
        model (SingleLayerSelfAttentionModel): Model instance.
        dataloader (DataLoader): Data loader.
        optimizer (optim.Optimizer): Optimizer.
        device (torch.device): Runtime device.
        stage (str): Train or eval stage.
        max_grad_norm (float): Gradient clipping value.

    Returns:
        Dict[str, float]: Aggregated metrics.
    """
    normalized_stage = stage.lower().strip()
    is_train = normalized_stage == "train"
    if is_train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    loop = tqdm(dataloader, desc = f"{normalized_stage.upper()}", leave = False)
    with torch.set_grad_enabled(is_train):
        for input_tokens, target_tokens, predict_mask in loop:
            input_tokens = input_tokens.to(device)
            target_tokens = target_tokens.to(device)
            predict_mask = predict_mask.to(device)

            logits, _ = model(input_tokens, padding_mask = None, return_scores = False)
            loss = compute_masked_loss(logits, target_tokens, predict_mask)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            correct_count, total_count = compute_masked_accuracy(logits, target_tokens, predict_mask)
            running_loss += float(loss.item())
            running_correct += correct_count
            running_total += total_count

            avg_loss = running_loss / max(len(loop), 1)
            avg_acc = 100.0 * running_correct / max(running_total, 1)
            loop.set_postfix(loss = f"{avg_loss:.4f}", acc = f"{avg_acc:.2f}%")

    epoch_loss = running_loss / max(len(dataloader), 1)
    epoch_acc = running_correct / max(running_total, 1)
    return {"loss": epoch_loss, "masked_acc": epoch_acc}


def collect_prediction_examples(
    model: SingleLayerSelfAttentionModel,
    dataset: Dataset,
    device: torch.device,
    num_examples: int
) -> List[Dict[str, List[int]]]:
    """
    Collect prediction samples.

    Args:
        model (SingleLayerSelfAttentionModel): Trained model.
        dataset (Dataset): Validation dataset or subset.
        device (torch.device): Runtime device.
        num_examples (int): Number of examples.

    Returns:
        List[Dict[str, List[int]]]: Prediction sample list.
    """
    model.eval()
    collator = MaskedCopyCollator(pad_token_id = PAD_TOKEN_ID)
    sampled = [dataset[idx] for idx in range(min(num_examples, len(dataset)))]
    input_tokens, target_tokens, predict_mask = collator(sampled)

    input_tokens = input_tokens.to(device)
    target_tokens = target_tokens.to(device)
    predict_mask = predict_mask.to(device)

    with torch.no_grad():
        logits, _ = model(input_tokens, padding_mask = None, return_scores = False)

    predictions = logits.argmax(dim = -1)
    examples = []
    for idx in range(input_tokens.size(0)):
        mask_positions = torch.where(predict_mask[idx])[0].detach().cpu().tolist()
        examples.append(
            {
                "input_tokens": input_tokens[idx].detach().cpu().tolist(),
                "target_tokens": target_tokens[idx].detach().cpu().tolist(),
                "predicted_tokens": predictions[idx].detach().cpu().tolist(),
                "masked_positions": mask_positions
            }
        )
    return examples


def save_json(path: str, payload: Dict) -> None:
    """
    Save dictionary to json file.

    Args:
        path (str): Output file path.
        payload (Dict): Json payload dictionary.

    Returns:
        None
    """
    os.makedirs(os.path.dirname(path), exist_ok = True)
    with open(path, "w", encoding = "utf-8") as file:
        json.dump(payload, file, ensure_ascii = False, indent = 2)


def main() -> None:
    """
    Training entry.

    Args:
        None

    Returns:
        None
    """
    args = parse_args()
    setup_seed(args.seed)
    device = get_device()

    os.makedirs(args.result_dir, exist_ok = True)
    os.makedirs(args.checkpoint_dir, exist_ok = True)

    logger.info("=" * 80)
    logger.info("Chapter 07 - Masked Copy with Single Self-Attention")
    logger.info("=" * 80)
    logger.info(f"Device: {device}")

    full_dataset = MaskedCopyDataset(
        num_samples = args.num_samples,
        vocab_size = args.vocab_size,
        min_seq_len = args.min_seq_len,
        max_seq_len = args.max_seq_len,
        mask_ratio = args.mask_ratio,
        seed = args.seed
    )
    train_size = int(len(full_dataset) * (1.0 - args.val_split))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    collator = MaskedCopyCollator(pad_token_id = PAD_TOKEN_ID)
    train_loader = DataLoader(
        train_dataset,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = args.num_workers,
        collate_fn = collator
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size = args.batch_size,
        shuffle = False,
        num_workers = args.num_workers,
        collate_fn = collator
    )

    model = SingleLayerSelfAttentionModel(
        vocab_size = args.vocab_size,
        d_model = args.d_model,
        pad_token_id = PAD_TOKEN_ID,
        dropout = args.dropout
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)

    history = []
    best_val_loss = float("inf")
    best_checkpoint_path = os.path.join(args.checkpoint_dir, "ch07_single_attn_best.pth")

    for epoch_idx in range(1, args.epochs + 1):
        logger.info("-" * 80)
        logger.info(f"Epoch {epoch_idx}/{args.epochs}")
        logger.info("-" * 80)

        train_metrics = run_epoch(
            model = model,
            dataloader = train_loader,
            optimizer = optimizer,
            device = device,
            stage = "train",
            max_grad_norm = args.max_grad_norm
        )
        val_metrics = run_epoch(
            model = model,
            dataloader = val_loader,
            optimizer = optimizer,
            device = device,
            stage = "eval",
            max_grad_norm = args.max_grad_norm
        )

        epoch_payload = {
            "epoch": epoch_idx,
            "train_loss": train_metrics["loss"],
            "train_masked_acc": train_metrics["masked_acc"],
            "val_loss": val_metrics["loss"],
            "val_masked_acc": val_metrics["masked_acc"]
        }
        history.append(epoch_payload)
        logger.info(f"Train loss: {train_metrics['loss']:.4f} | Train masked acc: {100.0 * train_metrics['masked_acc']:.2f}%")
        logger.info(f"Val   loss: {val_metrics['loss']:.4f} | Val   masked acc: {100.0 * val_metrics['masked_acc']:.2f}%")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), best_checkpoint_path)
            logger.info(f"Saved best checkpoint to: {best_checkpoint_path}")

    prediction_examples = collect_prediction_examples(
        model = model,
        dataset = val_dataset,
        device = device,
        num_examples = args.num_prediction_examples
    )

    metrics_path = os.path.join(args.result_dir, "metrics.json")
    predictions_path = os.path.join(args.result_dir, "predictions.json")
    run_config_path = os.path.join(args.result_dir, "run_config.json")

    save_json(metrics_path, {"history": history, "best_val_loss": best_val_loss})
    save_json(predictions_path, {"examples": prediction_examples})
    save_json(
        run_config_path,
        {
            "args": vars(args),
            "pad_token_id": PAD_TOKEN_ID,
            "mask_token_id": MASK_TOKEN_ID,
            "best_checkpoint": best_checkpoint_path,
            "best_val_loss": best_val_loss
        }
    )

    logger.info("=" * 80)
    logger.info("Chapter 07 training completed")
    logger.info(f"Saved metrics: {metrics_path}")
    logger.info(f"Saved predictions: {predictions_path}")
    logger.info(f"Saved config: {run_config_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers = [logging.StreamHandler()]
    )
    main()
