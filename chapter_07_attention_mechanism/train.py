import os
import sys
import logging
import argparse
from typing import Dict
from typing import List
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from tqdm import tqdm

sys.path.append(os.getcwd())

from chapter_07_attention_mechanism.dataset import PAD_TOKEN_ID
from chapter_07_attention_mechanism.dataset import BOS_TOKEN_ID
from chapter_07_attention_mechanism.dataset import EOS_TOKEN_ID
from chapter_07_attention_mechanism.dataset import ToyCopyDataset
from chapter_07_attention_mechanism.dataset import ToyCopyCollator
from chapter_07_attention_mechanism.masks import build_causal_mask
from chapter_07_attention_mechanism.masks import build_padding_mask
from chapter_07_attention_mechanism.transformer import Seq2SeqTransformer
from utils import get_device
from utils import save_json
from utils import setup_seed

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for chapter 07 training.

    Args:
        None

    Returns:
        argparse.Namespace: Parsed runtime arguments.
    """
    parser = argparse.ArgumentParser(description = "Chapter 07: Train Transformer on Toy Copy Task")
    parser.add_argument(
        "--seed",
        type = int,
        default = 42,
        help = "Random seed"
    )
    parser.add_argument(
        "--epochs",
        type = int,
        default = 10,
        help = "Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type = int,
        default = 64,
        help = "Batch size"
    )
    parser.add_argument(
        "--num_samples",
        type = int,
        default = 10000,
        help = "Number of synthetic samples"
    )
    parser.add_argument(
        "--val_split",
        type = float,
        default = 0.2,
        help = "Validation split ratio"
    )
    parser.add_argument(
        "--vocab_size",
        type = int,
        default = 64,
        help = "Vocabulary size including special tokens"
    )
    parser.add_argument(
        "--min_seq_len",
        type = int,
        default = 4,
        help = "Minimum sequence length"
    )
    parser.add_argument(
        "--max_seq_len",
        type = int,
        default = 20,
        help = "Maximum sequence length"
    )
    parser.add_argument(
        "--d_model",
        type = int,
        default = 128,
        help = "Model hidden dimension"
    )
    parser.add_argument(
        "--num_heads",
        type = int,
        default = 8,
        help = "Number of attention heads"
    )
    parser.add_argument(
        "--num_encoder_layers",
        type = int,
        default = 3,
        help = "Number of encoder layers"
    )
    parser.add_argument(
        "--num_decoder_layers",
        type = int,
        default = 3,
        help = "Number of decoder layers"
    )
    parser.add_argument(
        "--ffn_hidden_dim",
        type = int,
        default = 512,
        help = "Feed-forward hidden dimension"
    )
    parser.add_argument(
        "--dropout",
        type = float,
        default = 0.1,
        help = "Dropout rate"
    )
    parser.add_argument(
        "--lr",
        type = float,
        default = 1e-3,
        help = "Learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type = float,
        default = 1e-4,
        help = "AdamW weight decay"
    )
    parser.add_argument(
        "--max_grad_norm",
        type = float,
        default = 1.0,
        help = "Gradient clipping max norm"
    )
    parser.add_argument(
        "--num_workers",
        type = int,
        default = 0,
        help = "DataLoader workers"
    )
    parser.add_argument(
        "--num_prediction_examples",
        type = int,
        default = 5,
        help = "Number of greedy decode examples to save"
    )
    parser.add_argument(
        "--shape_check_only",
        action = "store_true",
        help = "Run shape and mask checks only"
    )
    parser.add_argument(
        "--result_dir",
        type = str,
        default = "chapter_07_attention_mechanism/results",
        help = "Directory to save metrics and predictions"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type = str,
        default = "chapter_07_attention_mechanism/checkpoints",
        help = "Directory to save model checkpoints"
    )
    return parser.parse_args()


def args_parser() -> argparse.Namespace:
    """
    Backward-compatible alias for parse_args.

    Args:
        None

    Returns:
        argparse.Namespace: Parsed runtime arguments.
    """
    return parse_args()


def compute_token_accuracy(logits: torch.Tensor, targets: torch.Tensor, pad_token_id: int) -> Tuple[int, int]:
    """
    Compute token-level accuracy while ignoring padded positions.

    Args:
        logits (torch.Tensor): Model logits with shape [batch, seq_len, vocab_size].
        targets (torch.Tensor): Target tokens with shape [batch, seq_len].
        pad_token_id (int): Padding token id.

    Returns:
        Tuple[int, int]: Number of correct tokens and valid tokens.
    """
    predictions = logits.argmax(dim = -1)
    valid_mask = targets.ne(pad_token_id)
    correct_mask = predictions.eq(targets) & valid_mask

    correct_count = int(correct_mask.sum().item())
    total_count = int(valid_mask.sum().item())
    return correct_count, total_count


def run_copy_epoch(
    model: Seq2SeqTransformer,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    pad_token_id: int,
    stage: str,
    epoch_idx: int,
    max_grad_norm: float
) -> Dict[str, float]:
    """
    Run one train or evaluation epoch for copy task.

    Args:
        model (Seq2SeqTransformer): Transformer model.
        dataloader (DataLoader): Input data loader.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer for training stage.
        device (torch.device): Runtime device.
        pad_token_id (int): Padding token id.
        stage (str): "train" or "eval".
        epoch_idx (int): Current epoch index.
        max_grad_norm (float): Gradient clipping max norm.

    Returns:
        Dict[str, float]: Aggregated loss and token accuracy.
    """
    normalized_stage = stage.lower().strip()
    if normalized_stage not in ["train", "eval"]:
        raise ValueError("stage must be 'train' or 'eval'.")

    is_train = normalized_stage == "train"
    if is_train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_correct = 0
    running_tokens = 0

    loop = tqdm(
        dataloader,
        desc = f"Epoch {epoch_idx} [{normalized_stage.upper()}]",
        leave = False
    )

    with torch.set_grad_enabled(is_train):
        for src_tokens, tgt_input_tokens, tgt_output_tokens in loop:
            src_tokens = src_tokens.to(device)
            tgt_input_tokens = tgt_input_tokens.to(device)
            tgt_output_tokens = tgt_output_tokens.to(device)

            logits = model(src_tokens, tgt_input_tokens)
            vocab_size = logits.size(-1)
            loss = criterion(logits.view(-1, vocab_size), tgt_output_tokens.view(-1))

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = max_grad_norm)
                optimizer.step()

            correct_count, token_count = compute_token_accuracy(logits, tgt_output_tokens, pad_token_id)
            running_correct += correct_count
            running_tokens += token_count
            running_loss += loss.item() * max(token_count, 1)

            avg_loss = running_loss / max(running_tokens, 1)
            avg_acc = 100.0 * running_correct / max(running_tokens, 1)
            loop.set_postfix(loss = f"{avg_loss:.4f}", token_acc = f"{avg_acc:.2f}%")

    epoch_loss = running_loss / max(running_tokens, 1)
    epoch_acc = 100.0 * running_correct / max(running_tokens, 1)
    return {
        "loss": epoch_loss,
        "token_acc": epoch_acc,
        "num_tokens": float(running_tokens)
    }


def trim_generated_tokens(token_ids: List[int], eos_token_id: int) -> List[int]:
    """
    Trim decoded tokens after EOS and remove prefix BOS if present.

    Args:
        token_ids (List[int]): Decoded token ids.
        eos_token_id (int): End-of-sequence token id.

    Returns:
        List[int]: Trimmed token ids.
    """
    trimmed = []
    start_index = 1 if len(token_ids) > 0 else 0

    for token_id in token_ids[start_index:]:
        trimmed.append(token_id)
        if token_id == eos_token_id:
            break
    return trimmed


def run_shape_and_mask_checks(model: Seq2SeqTransformer, device: torch.device) -> None:
    """
    Validate key module outputs and masks.

    Args:
        model (Seq2SeqTransformer): Transformer model.
        device (torch.device): Runtime device.

    Returns:
        None
    """
    logger.info("Running shape and mask checks...")
    model.eval()

    src_tokens = torch.tensor(
        [
            [7, 8, 9, EOS_TOKEN_ID, PAD_TOKEN_ID],
            [4, 5, EOS_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID]
        ],
        dtype = torch.long,
        device = device
    )
    tgt_input_tokens = torch.tensor(
        [
            [BOS_TOKEN_ID, 7, 8, 9, PAD_TOKEN_ID],
            [BOS_TOKEN_ID, 4, 5, PAD_TOKEN_ID, PAD_TOKEN_ID]
        ],
        dtype = torch.long,
        device = device
    )

    with torch.no_grad():
        logits, attention_dict = model(src_tokens, tgt_input_tokens, return_attn = True)

    expected_shape = (src_tokens.size(0), tgt_input_tokens.size(1), model.vocab_size)
    if logits.shape != expected_shape:
        raise RuntimeError(f"Unexpected logits shape: {logits.shape}, expected: {expected_shape}")

    src_mask = build_padding_mask(src_tokens, PAD_TOKEN_ID)
    if src_mask.dtype != torch.bool or src_mask.shape != (2, 1, 1, 5):
        raise RuntimeError("Padding mask check failed.")

    causal_mask = build_causal_mask(seq_len = 5, device = device)
    if not bool(causal_mask[0, 0, 0, 1].item()):
        raise RuntimeError("Causal mask future position should be masked.")
    if bool(causal_mask[0, 0, 2, 2].item()):
        raise RuntimeError("Causal mask diagonal should not be masked.")

    if len(attention_dict["encoder_self_attn"]) != len(model.encoder_layers):
        raise RuntimeError("Encoder attention map count mismatch.")
    if len(attention_dict["decoder_self_attn"]) != len(model.decoder_layers):
        raise RuntimeError("Decoder self-attention map count mismatch.")
    if len(attention_dict["decoder_cross_attn"]) != len(model.decoder_layers):
        raise RuntimeError("Decoder cross-attention map count mismatch.")

    logger.info("Shape and mask checks passed.")


def save_prediction_examples(
    model: Seq2SeqTransformer,
    dataset: ToyCopyDataset,
    device: torch.device,
    output_path: str,
    num_examples: int,
    max_target_len: int
) -> None:
    """
    Save greedy decoding examples to JSON.

    Args:
        model (Seq2SeqTransformer): Trained Transformer model.
        dataset (ToyCopyDataset): Source dataset.
        device (torch.device): Runtime device.
        output_path (str): Output json path.
        num_examples (int): Number of examples to export.
        max_target_len (int): Maximum target decode length.

    Returns:
        None
    """
    model.eval()
    examples = []
    limit = min(num_examples, len(dataset))

    with torch.no_grad():
        for index in range(limit):
            raw_tokens = dataset[index]
            src_tokens = raw_tokens + [EOS_TOKEN_ID]
            src_tensor = torch.tensor([src_tokens], dtype = torch.long, device = device)

            generated = model.greedy_decode(
                src_tokens = src_tensor,
                max_len = max_target_len,
                bos_token_id = BOS_TOKEN_ID,
                eos_token_id = EOS_TOKEN_ID
            )
            prediction_tokens = trim_generated_tokens(generated[0].tolist(), EOS_TOKEN_ID)

            examples.append(
                {
                    "index": index,
                    "source": src_tokens,
                    "target": raw_tokens + [EOS_TOKEN_ID],
                    "prediction": prediction_tokens
                }
            )

    save_json(examples, output_path)
    logger.info(f"Saved prediction examples to {output_path}")


def main() -> None:
    """
    Main training entry for chapter 07.

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

    logger.info(f"Device: {device}")
    logger.info(f"Arguments: {vars(args)}")

    dataset = ToyCopyDataset(
        num_samples = args.num_samples,
        vocab_size = args.vocab_size,
        min_seq_len = args.min_seq_len,
        max_seq_len = args.max_seq_len,
        seed = args.seed
    )

    train_size = int((1.0 - args.val_split) * len(dataset))
    val_size = len(dataset) - train_size
    split_generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator = split_generator)

    collator = ToyCopyCollator()
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

    model = Seq2SeqTransformer(
        vocab_size = args.vocab_size,
        d_model = args.d_model,
        num_heads = args.num_heads,
        num_encoder_layers = args.num_encoder_layers,
        num_decoder_layers = args.num_decoder_layers,
        ffn_hidden_dim = args.ffn_hidden_dim,
        dropout = args.dropout,
        max_len = args.max_seq_len + 8,
        pad_token_id = PAD_TOKEN_ID
    ).to(device)

    run_shape_and_mask_checks(model, device)
    if args.shape_check_only:
        logger.info("Shape check only mode completed.")
        return

    criterion = nn.CrossEntropyLoss(ignore_index = PAD_TOKEN_ID)
    optimizer = optim.AdamW(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)

    metrics = {
        "train_loss": [],
        "val_loss": [],
        "train_token_acc": [],
        "val_token_acc": []
    }

    best_val_loss = float("inf")
    best_checkpoint_path = os.path.join(args.checkpoint_dir, "transformer_copy_best.pth")

    for epoch_idx in range(1, args.epochs + 1):
        train_metrics = run_copy_epoch(
            model = model,
            dataloader = train_loader,
            criterion = criterion,
            optimizer = optimizer,
            device = device,
            pad_token_id = PAD_TOKEN_ID,
            stage = "train",
            epoch_idx = epoch_idx,
            max_grad_norm = args.max_grad_norm
        )
        val_metrics = run_copy_epoch(
            model = model,
            dataloader = val_loader,
            criterion = criterion,
            optimizer = optimizer,
            device = device,
            pad_token_id = PAD_TOKEN_ID,
            stage = "eval",
            epoch_idx = epoch_idx,
            max_grad_norm = args.max_grad_norm
        )

        metrics["train_loss"].append(train_metrics["loss"])
        metrics["val_loss"].append(val_metrics["loss"])
        metrics["train_token_acc"].append(train_metrics["token_acc"])
        metrics["val_token_acc"].append(val_metrics["token_acc"])

        logger.info(
            f"Epoch {epoch_idx}/{args.epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} Token Acc: {train_metrics['token_acc']:.2f}% | "
            f"Val Loss: {val_metrics['loss']:.4f} Token Acc: {val_metrics['token_acc']:.2f}%"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), best_checkpoint_path)
            logger.info(f"Saved new best checkpoint: {best_checkpoint_path}")

    metrics_path = os.path.join(args.result_dir, "metrics.json")
    save_json(metrics, metrics_path)

    run_config = vars(args).copy()
    run_config["best_val_loss"] = float(best_val_loss)
    run_config["best_checkpoint"] = best_checkpoint_path
    run_config_path = os.path.join(args.result_dir, "run_config.json")
    save_json(run_config, run_config_path)

    prediction_path = os.path.join(args.result_dir, "predictions.json")
    save_prediction_examples(
        model = model,
        dataset = dataset,
        device = device,
        output_path = prediction_path,
        num_examples = args.num_prediction_examples,
        max_target_len = args.max_seq_len + 4
    )

    logger.info(f"Saved metrics to {metrics_path}")
    logger.info(f"Saved run config to {run_config_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers = [logging.StreamHandler()]
    )
    main()
