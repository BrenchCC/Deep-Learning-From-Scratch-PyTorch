import os
import sys
import json
import logging
import argparse
from typing import Dict, Any

import torch

# Add project root to Python path
sys.path.append(os.getcwd())

from chapter_07_attention_mechanism.attention import ScaledDotProductAttention
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
    parser = argparse.ArgumentParser(description = "Chapter 07 Attention Demo")
    parser.add_argument("--batch_size", type = int, default = 2, help = "Batch size")
    parser.add_argument("--seq_len", type = int, default = 6, help = "Sequence length")
    parser.add_argument("--dim", type = int, default = 8, help = "Vector dimension")
    parser.add_argument("--seed", type = int, default = 42, help = "Random seed")
    parser.add_argument(
        "--result_path",
        type = str,
        default = "chapter_07_attention_mechanism/results/attention_demo.json",
        help = "Result json path"
    )
    return parser.parse_args()


def build_demo_payload(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scores: torch.Tensor,
    context: torch.Tensor
) -> Dict[str, Any]:
    """
    Build serializable demo payload.

    Args:
        query (torch.Tensor): Query tensor.
        key (torch.Tensor): Key tensor.
        value (torch.Tensor): Value tensor.
        scores (torch.Tensor): Attention score tensor.
        context (torch.Tensor): Context tensor.

    Returns:
        Dict[str, Any]: Json payload dictionary.
    """
    payload = {
        "query_shape": list(query.shape),
        "key_shape": list(key.shape),
        "value_shape": list(value.shape),
        "attention_score_shape": list(scores.shape),
        "context_shape": list(context.shape),
        "attention_score_sample": scores[0].detach().cpu().tolist(),
        "context_sample": context[0].detach().cpu().tolist()
    }
    return payload


def main() -> None:
    """
    Run attention mechanism demo.

    Args:
        None

    Returns:
        None
    """
    args = parse_args()
    setup_seed(args.seed)

    query = torch.randn(args.batch_size, args.seq_len, args.dim)
    key = torch.randn(args.batch_size, args.seq_len, args.dim)
    value = torch.randn(args.batch_size, args.seq_len, args.dim)

    attention = ScaledDotProductAttention(dropout = 0.0)
    context, scores = attention(query, key, value, mask = None)

    payload = build_demo_payload(query, key, value, scores, context)

    os.makedirs(os.path.dirname(args.result_path), exist_ok = True)
    with open(args.result_path, "w", encoding = "utf-8") as file:
        json.dump(payload, file, ensure_ascii = False, indent = 2)

    logger.info("=" * 80)
    logger.info("Chapter 07 attention demo completed")
    logger.info(f"Saved demo output to: {args.result_path}")
    logger.info(f"Attention score shape: {payload['attention_score_shape']}")
    logger.info(f"Context shape: {payload['context_shape']}")
    logger.info("=" * 80)


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers = [logging.StreamHandler()]
    )
    main()
