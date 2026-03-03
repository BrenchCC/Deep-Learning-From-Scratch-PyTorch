import os
import sys
import json
import logging
import argparse
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from tqdm import tqdm

# Add project root to Python path
sys.path.append(os.getcwd())

from chapter_08_transformer_vanilla.dataset import PAD_TOKEN_ID
from chapter_08_transformer_vanilla.dataset import BOS_TOKEN_ID
from chapter_08_transformer_vanilla.dataset import EOS_TOKEN_ID
from chapter_08_transformer_vanilla.dataset import SortDataset
from chapter_08_transformer_vanilla.dataset import ToyTranslationDataset
from chapter_08_transformer_vanilla.dataset import Seq2SeqCollator
from chapter_08_transformer_vanilla.transformer import VanillaTransformer
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
    parser = argparse.ArgumentParser(description = "Chapter 08 Vanilla Transformer Training")
    parser.add_argument("--task", type = str, choices = ["sort", "translate"], default = "sort", help = "Task")
    parser.add_argument("--seed", type = int, default = 42, help = "Random seed")
    parser.add_argument("--epochs", type = int, default = 10, help = "Training epochs")
    parser.add_argument("--batch_size", type = int, default = 64, help = "Batch size")
    parser.add_argument("--val_split", type = float, default = 0.2, help = "Validation split")
    parser.add_argument("--num_workers", type = int, default = 0, help = "DataLoader workers")

    parser.add_argument("--num_samples", type = int, default = 10000, help = "Sort dataset size")
    parser.add_argument("--vocab_size", type = int, default = 64, help = "Sort task vocabulary size")
    parser.add_argument("--min_seq_len", type = int, default = 4, help = "Min sequence length")
    parser.add_argument("--max_seq_len", type = int, default = 12, help = "Max sequence length")

    parser.add_argument(
        "--translation_data_path",
        type = str,
        default = "chapter_08_transformer_vanilla/data/toy_translation_pairs.tsv",
        help = "Local translation pair file"
    )
    parser.add_argument("--translation_repeat_factor", type = int, default = 40, help = "Repeat factor")

    parser.add_argument("--d_model", type = int, default = 128, help = "Model hidden size")
    parser.add_argument("--num_heads", type = int, default = 8, help = "Attention heads")
    parser.add_argument("--num_encoder_layers", type = int, default = 2, help = "Encoder layers")
    parser.add_argument("--num_decoder_layers", type = int, default = 2, help = "Decoder layers")
    parser.add_argument("--ffn_hidden_dim", type = int, default = 256, help = "FFN hidden size")
    parser.add_argument("--dropout", type = float, default = 0.1, help = "Dropout")
    parser.add_argument("--lr", type = float, default = 1e-3, help = "Learning rate")
    parser.add_argument("--weight_decay", type = float, default = 1e-4, help = "Weight decay")
    parser.add_argument("--max_grad_norm", type = float, default = 1.0, help = "Grad clip")

    parser.add_argument("--num_prediction_examples", type = int, default = 5, help = "Prediction example count")
    parser.add_argument(
        "--result_dir",
        type = str,
        default = "chapter_08_transformer_vanilla/results",
        help = "Result folder"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type = str,
        default = "chapter_08_transformer_vanilla/checkpoints",
        help = "Checkpoint folder"
    )
    return parser.parse_args()


def compute_token_accuracy(logits: torch.Tensor, targets: torch.Tensor, pad_token_id: int) -> Tuple[int, int]:
    """
    Compute non-padding token accuracy.

    Args:
        logits (torch.Tensor): Model logits [batch, seq_len, vocab_size].
        targets (torch.Tensor): Target ids [batch, seq_len].
        pad_token_id (int): Padding token id.

    Returns:
        Tuple[int, int]: Correct and total count.
    """
    predictions = logits.argmax(dim = -1)
    valid_mask = targets.ne(pad_token_id)
    correct_mask = predictions.eq(targets) & valid_mask
    return int(correct_mask.sum().item()), int(valid_mask.sum().item())


def run_epoch(
    model: VanillaTransformer,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    stage: str,
    pad_token_id: int,
    max_grad_norm: float
) -> Dict[str, float]:
    """
    Run one train or eval epoch.

    Args:
        model (VanillaTransformer): Model instance.
        dataloader (DataLoader): DataLoader.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer.
        device (torch.device): Runtime device.
        stage (str): Stage name.
        pad_token_id (int): Padding token id.
        max_grad_norm (float): Gradient clipping norm.

    Returns:
        Dict[str, float]: Epoch metrics.
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
        for src_tokens, tgt_input_tokens, tgt_output_tokens in loop:
            src_tokens = src_tokens.to(device)
            tgt_input_tokens = tgt_input_tokens.to(device)
            tgt_output_tokens = tgt_output_tokens.to(device)

            logits = model(src_tokens, tgt_input_tokens, return_attn = False)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output_tokens.reshape(-1))

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            correct_count, total_count = compute_token_accuracy(logits, tgt_output_tokens, pad_token_id)

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


def _trim_special_ids(ids: List[int]) -> List[int]:
    """
    Remove special ids for display.

    Args:
        ids (List[int]): Token id list.

    Returns:
        List[int]: Trimmed token ids.
    """
    output = []
    for idx in ids:
        if idx == PAD_TOKEN_ID:
            continue
        if idx == BOS_TOKEN_ID:
            continue
        if idx == EOS_TOKEN_ID:
            break
        output.append(idx)
    return output


def _decode_for_display(ids: List[int], id_to_token: Optional[Dict[int, str]]) -> List:
    """
    Decode ids for display.

    Args:
        ids (List[int]): Id list.
        id_to_token (Optional[Dict[int, str]]): Optional vocab map.

    Returns:
        List: Decoded representation.
    """
    trimmed = _trim_special_ids(ids)
    if id_to_token is None:
        return trimmed
    return [id_to_token.get(idx, "<UNK>") for idx in trimmed]


def collect_prediction_examples(
    model: VanillaTransformer,
    dataloader: DataLoader,
    device: torch.device,
    max_examples: int,
    id_to_token: Optional[Dict[int, str]] = None
) -> List[Dict[str, List]]:
    """
    Collect prediction examples from a dataloader.

    Args:
        model (VanillaTransformer): Trained model.
        dataloader (DataLoader): Validation dataloader.
        device (torch.device): Runtime device.
        max_examples (int): Number of examples.
        id_to_token (Optional[Dict[int, str]]): Optional decode table.

    Returns:
        List[Dict[str, List]]: Prediction examples.
    """
    model.eval()
    examples = []

    with torch.no_grad():
        for src_tokens, _, tgt_output_tokens in dataloader:
            src_tokens = src_tokens.to(device)
            tgt_output_tokens = tgt_output_tokens.to(device)

            generated = model.greedy_decode(
                src_tokens,
                max_len = tgt_output_tokens.size(1) + 4,
                bos_token_id = BOS_TOKEN_ID,
                eos_token_id = EOS_TOKEN_ID
            )

            for idx in range(src_tokens.size(0)):
                src_ids = src_tokens[idx].detach().cpu().tolist()
                tgt_ids = tgt_output_tokens[idx].detach().cpu().tolist()
                pred_ids = generated[idx].detach().cpu().tolist()

                examples.append(
                    {
                        "src": _decode_for_display(src_ids, id_to_token),
                        "target": _decode_for_display(tgt_ids, id_to_token),
                        "prediction": _decode_for_display(pred_ids, id_to_token)
                    }
                )
                if len(examples) >= max_examples:
                    return examples

    return examples


def save_json(path: str, payload: Dict) -> None:
    """
    Save payload as json.

    Args:
        path (str): Output file path.
        payload (Dict): Json payload.

    Returns:
        None
    """
    os.makedirs(os.path.dirname(path), exist_ok = True)
    with open(path, "w", encoding = "utf-8") as file:
        json.dump(payload, file, ensure_ascii = False, indent = 2)


def build_task_dataset(args: argparse.Namespace):
    """
    Build task dataset.

    Args:
        args (argparse.Namespace): Runtime arguments.

    Returns:
        tuple: dataset object and optional id_to_token map.
    """
    if args.task == "sort":
        dataset = SortDataset(
            num_samples = args.num_samples,
            vocab_size = args.vocab_size,
            min_seq_len = args.min_seq_len,
            max_seq_len = args.max_seq_len,
            seed = args.seed
        )
        return dataset, None

    dataset = ToyTranslationDataset(
        data_path = args.translation_data_path,
        repeat_factor = args.translation_repeat_factor
    )
    return dataset, dataset.id_to_token


def get_vocab_size(args: argparse.Namespace, dataset) -> int:
    """
    Get task-specific vocabulary size.

    Args:
        args (argparse.Namespace): Runtime arguments.
        dataset: Dataset object.

    Returns:
        int: Vocabulary size.
    """
    if args.task == "sort":
        return args.vocab_size
    return dataset.vocab_size


def train_main() -> None:
    """
    Main training entry.

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
    logger.info(f"Chapter 08 - Vanilla Transformer Task: {args.task}")
    logger.info("=" * 80)
    logger.info(f"Device: {device}")

    dataset, id_to_token = build_task_dataset(args)

    train_size = int(len(dataset) * (1.0 - args.val_split))
    val_size = len(dataset) - train_size
    split_generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator = split_generator)

    collator = Seq2SeqCollator(pad_token_id = PAD_TOKEN_ID)
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

    vocab_size = get_vocab_size(args, dataset)
    model = VanillaTransformer(
        vocab_size = vocab_size,
        d_model = args.d_model,
        num_heads = args.num_heads,
        num_encoder_layers = args.num_encoder_layers,
        num_decoder_layers = args.num_decoder_layers,
        ffn_hidden_dim = args.ffn_hidden_dim,
        dropout = args.dropout,
        pad_token_id = PAD_TOKEN_ID
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index = PAD_TOKEN_ID)
    optimizer = optim.AdamW(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)

    metrics_history = []
    best_val_loss = float("inf")

    if args.task == "sort":
        checkpoint_path = os.path.join(args.checkpoint_dir, "transformer_sort_best.pth")
        metrics_path = os.path.join(args.result_dir, "sort_metrics.json")
        predictions_path = os.path.join(args.result_dir, "sort_predictions.json")
    else:
        checkpoint_path = os.path.join(args.checkpoint_dir, "transformer_translate_best.pth")
        metrics_path = os.path.join(args.result_dir, "translate_metrics.json")
        predictions_path = os.path.join(args.result_dir, "translate_predictions.json")

    for epoch_idx in range(1, args.epochs + 1):
        logger.info("-" * 80)
        logger.info(f"Epoch {epoch_idx}/{args.epochs}")
        logger.info("-" * 80)

        train_metrics = run_epoch(
            model = model,
            dataloader = train_loader,
            criterion = criterion,
            optimizer = optimizer,
            device = device,
            stage = "train",
            pad_token_id = PAD_TOKEN_ID,
            max_grad_norm = args.max_grad_norm
        )
        val_metrics = run_epoch(
            model = model,
            dataloader = val_loader,
            criterion = criterion,
            optimizer = optimizer,
            device = device,
            stage = "eval",
            pad_token_id = PAD_TOKEN_ID,
            max_grad_norm = args.max_grad_norm
        )

        epoch_payload = {
            "epoch": epoch_idx,
            "train_loss": train_metrics["loss"],
            "train_token_acc": train_metrics["token_acc"],
            "val_loss": val_metrics["loss"],
            "val_token_acc": val_metrics["token_acc"]
        }
        metrics_history.append(epoch_payload)

        logger.info(f"Train loss: {train_metrics['loss']:.4f} | Train acc: {100.0 * train_metrics['token_acc']:.2f}%")
        logger.info(f"Val   loss: {val_metrics['loss']:.4f} | Val   acc: {100.0 * val_metrics['token_acc']:.2f}%")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Saved best checkpoint: {checkpoint_path}")

    prediction_examples = collect_prediction_examples(
        model = model,
        dataloader = val_loader,
        device = device,
        max_examples = args.num_prediction_examples,
        id_to_token = id_to_token
    )

    save_json(metrics_path, {"history": metrics_history, "best_val_loss": best_val_loss})
    save_json(predictions_path, {"examples": prediction_examples})

    run_config_path = os.path.join(args.result_dir, "run_config.json")
    save_json(
        run_config_path,
        {
            "task": args.task,
            "args": vars(args),
            "pad_token_id": PAD_TOKEN_ID,
            "bos_token_id": BOS_TOKEN_ID,
            "eos_token_id": EOS_TOKEN_ID,
            "vocab_size": vocab_size,
            "best_checkpoint": checkpoint_path,
            "best_val_loss": best_val_loss
        }
    )

    logger.info("=" * 80)
    logger.info("Chapter 08 training completed")
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
    train_main()
