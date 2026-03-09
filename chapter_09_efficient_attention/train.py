import os
import sys
import json
import logging
import argparse
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Add project root to Python path
sys.path.append(os.getcwd())

from chapter_09_efficient_attention.dataset import PAD_TOKEN_ID, ToyNextTokenDataset, ToyNextTokenCollator
from chapter_09_efficient_attention.common import AttentionConfig, variant_kv_channels, estimate_kv_cache_bytes, format_bytes_as_megabytes
from chapter_09_efficient_attention.model import EfficientAttentionLM
from utils import get_device, setup_seed

logger = logging.getLogger(__name__)


VARIANTS = ["mha", "mqa", "gqa", "mla"]


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Args:
        None

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description = "Chapter 09 Efficient Attention Training")
    parser.add_argument("--variant", type = str, choices = VARIANTS + ["all"], default = "all", help = "Attention variant")

    parser.add_argument("--seed", type = int, default = 42, help = "Random seed")
    parser.add_argument("--epochs", type = int, default = 1, help = "Training epochs")
    parser.add_argument("--batch_size", type = int, default = 64, help = "Batch size")
    parser.add_argument("--num_workers", type = int, default = 0, help = "DataLoader workers")
    parser.add_argument("--val_split", type = float, default = 0.2, help = "Validation split")

    parser.add_argument("--num_samples", type = int, default = 2000, help = "Dataset sample size")
    parser.add_argument("--vocab_size", type = int, default = 128, help = "Vocabulary size")
    parser.add_argument("--seq_len", type = int, default = 64, help = "Sequence length")

    parser.add_argument("--d_model", type = int, default = 512, help = "Hidden size")
    parser.add_argument("--num_heads", type = int, default = 8, help = "Number of query heads")
    parser.add_argument("--num_kv_heads", type = int, default = 2, help = "Number of KV heads for GQA")
    parser.add_argument("--latent_dim", type = int, default = 64, help = "Latent dimension for MLA")
    parser.add_argument("--dropout", type = float, default = 0.1, help = "Dropout")

    parser.add_argument("--lr", type = float, default = 1e-3, help = "Learning rate")
    parser.add_argument("--weight_decay", type = float, default = 1e-4, help = "Weight decay")
    parser.add_argument("--max_grad_norm", type = float, default = 1.0, help = "Gradient clipping")

    parser.add_argument("--num_prediction_examples", type = int, default = 5, help = "Prediction sample count")
    parser.add_argument(
        "--result_dir",
        type = str,
        default = "chapter_09_efficient_attention/results",
        help = "Directory for result files"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type = str,
        default = "chapter_09_efficient_attention/checkpoints",
        help = "Directory for checkpoints"
    )
    return parser.parse_args()


def save_json(path: str, payload: Dict) -> None:
    """
    Save payload as JSON.

    Args:
        path (str): Output file path.
        payload (Dict): JSON payload.

    Returns:
        None
    """
    os.makedirs(os.path.dirname(path), exist_ok = True)
    with open(path, "w", encoding = "utf-8") as file:
        json.dump(payload, file, ensure_ascii = False, indent = 2)


def compute_token_accuracy(logits: torch.Tensor, labels: torch.Tensor, pad_token_id: int) -> Tuple[int, int]:
    """
    Compute non-padding token accuracy.

    Args:
        logits (torch.Tensor): Logits [B, S, V].
        labels (torch.Tensor): Labels [B, S].
        pad_token_id (int): Padding token id.

    Returns:
        Tuple[int, int]: Correct count and valid token count.
    """
    predictions = logits.argmax(dim = -1)
    valid_mask = labels.ne(pad_token_id)
    correct_mask = predictions.eq(labels) & valid_mask
    return int(correct_mask.sum().item()), int(valid_mask.sum().item())


def run_epoch(
    model: EfficientAttentionLM,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    stage: str,
    max_grad_norm: float,
    pad_token_id: int
) -> Dict[str, float]:
    """
    Run one train or eval epoch.

    Args:
        model (EfficientAttentionLM): Model instance.
        dataloader (DataLoader): DataLoader object.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer.
        device (torch.device): Runtime device.
        stage (str): Stage name.
        max_grad_norm (float): Gradient clipping norm.
        pad_token_id (int): Padding token id.

    Returns:
        Dict[str, float]: Loss and token accuracy.
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
    batch_count = 0

    loop = tqdm(dataloader, desc = normalized_stage.upper(), leave = False)

    with torch.set_grad_enabled(is_train):
        for input_ids, labels, padding_mask in loop:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            padding_mask = padding_mask.to(device)

            logits, _ = model(input_ids = input_ids, padding_mask = padding_mask, need_weights = False)
            loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            correct_count, total_count = compute_token_accuracy(logits, labels, pad_token_id)
            running_loss += float(loss.item())
            running_correct += correct_count
            running_total += total_count
            batch_count += 1

            loop.set_postfix(
                loss = f"{running_loss / max(batch_count, 1):.4f}",
                acc = f"{100.0 * running_correct / max(running_total, 1):.2f}%"
            )

    return {
        "loss": running_loss / max(batch_count, 1),
        "token_acc": running_correct / max(running_total, 1)
    }


def collect_prediction_examples(
    model: EfficientAttentionLM,
    dataloader: DataLoader,
    device: torch.device,
    max_examples: int
) -> List[Dict[str, List[int]]]:
    """
    Collect prediction examples from validation loader.

    Args:
        model (EfficientAttentionLM): Trained model.
        dataloader (DataLoader): Validation loader.
        device (torch.device): Runtime device.
        max_examples (int): Number of examples to collect.

    Returns:
        List[Dict[str, List[int]]]: Prediction sample list.
    """
    model.eval()
    examples = []

    with torch.no_grad():
        for input_ids, labels, padding_mask in dataloader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            padding_mask = padding_mask.to(device)

            logits, _ = model(input_ids = input_ids, padding_mask = padding_mask, need_weights = False)
            predictions = logits.argmax(dim = -1)

            for index in range(input_ids.size(0)):
                examples.append(
                    {
                        "input_ids": input_ids[index].detach().cpu().tolist(),
                        "target_ids": labels[index].detach().cpu().tolist(),
                        "predicted_ids": predictions[index].detach().cpu().tolist()
                    }
                )
                if len(examples) >= max_examples:
                    return examples

    return examples


def build_dataloaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation dataloaders.

    Args:
        args (argparse.Namespace): Runtime arguments.

    Returns:
        Tuple[DataLoader, DataLoader]: Train and validation dataloaders.
    """
    dataset = ToyNextTokenDataset(
        num_samples = args.num_samples,
        vocab_size = args.vocab_size,
        seq_len = args.seq_len,
        seed = args.seed
    )

    train_size = int(len(dataset) * (1.0 - args.val_split))
    val_size = len(dataset) - train_size
    split_generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator = split_generator)

    collator = ToyNextTokenCollator(pad_token_id = PAD_TOKEN_ID)

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

    return train_loader, val_loader


def train_single_variant(
    variant: str,
    args: argparse.Namespace,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """
    Train one attention variant.

    Args:
        variant (str): Variant name.
        args (argparse.Namespace): Runtime arguments.
        train_loader (DataLoader): Train dataloader.
        val_loader (DataLoader): Validation dataloader.
        device (torch.device): Runtime device.

    Returns:
        Dict[str, float]: Summary metrics.
    """
    config = AttentionConfig(
        d_model = args.d_model,
        num_heads = args.num_heads,
        num_kv_heads = args.num_kv_heads,
        latent_dim = args.latent_dim,
        dropout = args.dropout
    )

    model = EfficientAttentionLM(
        vocab_size = args.vocab_size,
        attention_variant = variant,
        config = config,
        pad_token_id = PAD_TOKEN_ID
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index = PAD_TOKEN_ID)
    optimizer = optim.AdamW(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)

    checkpoint_path = os.path.join(args.checkpoint_dir, f"{variant}_best.pth")
    metrics_path = os.path.join(args.result_dir, f"metrics_{variant}.json")
    predictions_path = os.path.join(args.result_dir, f"predictions_{variant}.json")

    metrics_history = []
    best_val_loss = float("inf")
    best_val_acc = 0.0

    for epoch_index in range(1, args.epochs + 1):
        logger.info("-" * 80)
        logger.info(f"Variant {variant.upper()} - Epoch {epoch_index}/{args.epochs}")
        logger.info("-" * 80)

        train_metrics = run_epoch(
            model = model,
            dataloader = train_loader,
            criterion = criterion,
            optimizer = optimizer,
            device = device,
            stage = "train",
            max_grad_norm = args.max_grad_norm,
            pad_token_id = PAD_TOKEN_ID
        )
        val_metrics = run_epoch(
            model = model,
            dataloader = val_loader,
            criterion = criterion,
            optimizer = optimizer,
            device = device,
            stage = "eval",
            max_grad_norm = args.max_grad_norm,
            pad_token_id = PAD_TOKEN_ID
        )

        history_entry = {
            "epoch": epoch_index,
            "train_loss": train_metrics["loss"],
            "train_token_acc": train_metrics["token_acc"],
            "val_loss": val_metrics["loss"],
            "val_token_acc": val_metrics["token_acc"]
        }
        metrics_history.append(history_entry)

        logger.info(
            f"Variant {variant.upper()} | Train loss: {train_metrics['loss']:.4f} | "
            f"Train acc: {100.0 * train_metrics['token_acc']:.2f}%"
        )
        logger.info(
            f"Variant {variant.upper()} | Val   loss: {val_metrics['loss']:.4f} | "
            f"Val   acc: {100.0 * val_metrics['token_acc']:.2f}%"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_val_acc = val_metrics["token_acc"]
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Saved best checkpoint: {checkpoint_path}")

    prediction_examples = collect_prediction_examples(
        model = model,
        dataloader = val_loader,
        device = device,
        max_examples = args.num_prediction_examples
    )

    save_json(
        metrics_path,
        {
            "variant": variant,
            "history": metrics_history,
            "best_val_loss": best_val_loss,
            "best_val_token_acc": best_val_acc
        }
    )
    save_json(predictions_path, {"variant": variant, "examples": prediction_examples})

    effective_seq_len = args.seq_len + 1
    kv_channels = variant_kv_channels(
        variant = variant,
        d_model = args.d_model,
        num_heads = args.num_heads,
        num_kv_heads = args.num_kv_heads,
        latent_dim = args.latent_dim
    )
    bytes_per_elem = torch.tensor(0.0, dtype = torch.float32).element_size()
    kv_cache_bytes = estimate_kv_cache_bytes(
        batch_size = args.batch_size,
        seq_len = effective_seq_len,
        kv_channels = kv_channels,
        bytes_per_elem = bytes_per_elem
    )

    return {
        "variant": variant,
        "best_val_loss": float(best_val_loss),
        "best_val_token_acc": float(best_val_acc),
        "metrics_path": metrics_path,
        "predictions_path": predictions_path,
        "checkpoint_path": checkpoint_path,
        "kv_channels_per_tensor": int(kv_channels),
        "kv_cache_bytes": int(kv_cache_bytes),
        "kv_cache_mb": format_bytes_as_megabytes(kv_cache_bytes)
    }


def train_main() -> None:
    """
    Main training entry for chapter 09.

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
    logger.info("Chapter 09 - Efficient Attention Training")
    logger.info("=" * 80)
    logger.info(f"Device: {device}")

    train_loader, val_loader = build_dataloaders(args)
    variant_list = VARIANTS if args.variant == "all" else [args.variant]

    summaries = []
    for variant in variant_list:
        summary = train_single_variant(
            variant = variant,
            args = args,
            train_loader = train_loader,
            val_loader = val_loader,
            device = device
        )
        summaries.append(summary)

    if len(summaries) > 0:
        mha_cache_bytes = next(
            item["kv_cache_bytes"]
            for item in summaries
            if item["variant"] == "mha"
        ) if any(item["variant"] == "mha" for item in summaries) else summaries[0]["kv_cache_bytes"]

        for item in summaries:
            item["compression_ratio_vs_mha"] = float(mha_cache_bytes / max(item["kv_cache_bytes"], 1))

    sorted_by_loss = sorted(summaries, key = lambda item: item["best_val_loss"])

    compare_path = os.path.join(args.result_dir, "attention_compare.json")
    compare_payload = {}
    if os.path.exists(compare_path):
        try:
            with open(compare_path, "r", encoding = "utf-8") as file:
                loaded_payload = json.load(file)
            if isinstance(loaded_payload, dict):
                if "demo" in loaded_payload:
                    compare_payload["demo"] = loaded_payload["demo"]
        except (json.JSONDecodeError, OSError):
            compare_payload = {}

    compare_payload["training"] = {
        "meta": {
            "variant_mode": args.variant,
            "device": str(device),
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "num_samples": args.num_samples,
            "seq_len": args.seq_len,
            "d_model": args.d_model,
            "num_heads": args.num_heads,
            "num_kv_heads": args.num_kv_heads,
            "latent_dim": args.latent_dim
        },
        "summaries": summaries,
        "ranking_by_val_loss": [item["variant"] for item in sorted_by_loss]
    }
    save_json(compare_path, compare_payload)

    run_config_path = os.path.join(args.result_dir, "run_config.json")
    save_json(
        run_config_path,
        {
            "args": vars(args),
            "variants_executed": variant_list,
            "compare_path": compare_path
        }
    )

    logger.info("=" * 80)
    logger.info("Chapter 09 training completed")
    logger.info(f"Saved aggregate compare: {compare_path}")
    logger.info(f"Saved run config: {run_config_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers = [logging.StreamHandler()]
    )
    train_main()
