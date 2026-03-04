import os
import sys
import json
import logging
import argparse
from typing import Dict
from typing import List
from typing import Any

import torch
import matplotlib.pyplot as plt

# Add project root to Python path
sys.path.append(os.getcwd())

from chapter_09_efficient_attention.common import AttentionConfig
from chapter_09_efficient_attention.common import format_bytes_as_megabytes
from chapter_09_efficient_attention.common import variant_kv_channels
from chapter_09_efficient_attention.common import estimate_kv_cache_bytes
from chapter_09_efficient_attention.model import build_attention_block
from utils import setup_seed

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
    parser = argparse.ArgumentParser(description = "Chapter 09 Efficient Attention Demo")
    parser.add_argument("--seed", type = int, default = 42, help = "Random seed")
    parser.add_argument("--batch_size", type = int, default = 2, help = "Batch size")
    parser.add_argument("--seq_len", type = int, default = 64, help = "Sequence length")
    parser.add_argument("--d_model", type = int, default = 512, help = "Hidden size")
    parser.add_argument("--num_heads", type = int, default = 8, help = "Number of query heads")
    parser.add_argument("--num_kv_heads", type = int, default = 2, help = "Number of KV heads for GQA")
    parser.add_argument("--latent_dim", type = int, default = 64, help = "Latent dimension for MLA")
    parser.add_argument("--dropout", type = float, default = 0.0, help = "Dropout")
    parser.add_argument(
        "--result_path",
        type = str,
        default = "chapter_09_efficient_attention/results/attention_compare.json",
        help = "Output json path"
    )
    parser.add_argument(
        "--image_dir",
        type = str,
        default = "chapter_09_efficient_attention/images",
        help = "Output image directory"
    )
    return parser.parse_args()


def _build_config(args: argparse.Namespace) -> AttentionConfig:
    """
    Build attention config from arguments.

    Args:
        args (argparse.Namespace): Runtime arguments.

    Returns:
        AttentionConfig: Config object.
    """
    return AttentionConfig(
        d_model = args.d_model,
        num_heads = args.num_heads,
        num_kv_heads = args.num_kv_heads,
        latent_dim = args.latent_dim,
        dropout = args.dropout
    )


def _compute_output_stats(output: torch.Tensor) -> Dict[str, Any]:
    """
    Compute summary stats for one output tensor.

    Args:
        output (torch.Tensor): Output tensor [B, S, D].

    Returns:
        Dict[str, Any]: Statistics dictionary.
    """
    return {
        "shape": list(output.shape),
        "mean": float(output.mean().item()),
        "std": float(output.std(unbiased = False).item()),
        "l2_norm": float(torch.norm(output).item())
    }


def _plot_kv_cache_comparison(variant_payloads: List[Dict[str, Any]], output_path: str) -> None:
    """
    Plot KV cache size comparison.

    Args:
        variant_payloads (List[Dict[str, Any]]): Variant result list.
        output_path (str): Target image path.

    Returns:
        None
    """
    variant_names = [payload["variant"].upper() for payload in variant_payloads]
    cache_sizes_mb = [payload["kv_cache_mb"] for payload in variant_payloads]

    plt.figure(figsize = (8, 5))
    bars = plt.bar(variant_names, cache_sizes_mb, color = ["#4472C4", "#ED7D31", "#70AD47", "#A5A5A5"])
    plt.ylabel("KV Cache (MB)")
    plt.title("Chapter 09 KV Cache Comparison")
    plt.grid(axis = "y", alpha = 0.25)

    for bar, value in zip(bars, cache_sizes_mb):
        plt.text(bar.get_x() + bar.get_width() * 0.5, value, f"{value:.3f}", ha = "center", va = "bottom")

    plt.tight_layout()
    plt.savefig(output_path, dpi = 200)
    plt.close()


def _plot_output_stats(variant_payloads: List[Dict[str, Any]], output_path: str) -> None:
    """
    Plot output stats for all variants.

    Args:
        variant_payloads (List[Dict[str, Any]]): Variant result list.
        output_path (str): Target image path.

    Returns:
        None
    """
    variant_names = [payload["variant"].upper() for payload in variant_payloads]
    means = [payload["output_stats"]["mean"] for payload in variant_payloads]
    stds = [payload["output_stats"]["std"] for payload in variant_payloads]
    norms = [payload["output_stats"]["l2_norm"] for payload in variant_payloads]

    fig, axes = plt.subplots(1, 3, figsize = (15, 4))

    axes[0].bar(variant_names, means, color = "#5B9BD5")
    axes[0].set_title("Output Mean")
    axes[0].grid(axis = "y", alpha = 0.25)

    axes[1].bar(variant_names, stds, color = "#9BBB59")
    axes[1].set_title("Output Std")
    axes[1].grid(axis = "y", alpha = 0.25)

    axes[2].bar(variant_names, norms, color = "#C0504D")
    axes[2].set_title("Output L2 Norm")
    axes[2].grid(axis = "y", alpha = 0.25)

    fig.suptitle("Chapter 09 Output Statistics Comparison")
    fig.tight_layout()
    fig.savefig(output_path, dpi = 200)
    plt.close(fig)


def main() -> None:
    """
    Run efficient attention demo.

    Args:
        None

    Returns:
        None
    """
    args = parse_args()
    setup_seed(args.seed)

    os.makedirs(os.path.dirname(args.result_path), exist_ok = True)
    os.makedirs(args.image_dir, exist_ok = True)

    config = _build_config(args)
    hidden_states = torch.randn(args.batch_size, args.seq_len, args.d_model)

    baseline_output = None
    variant_payloads = []
    bytes_per_elem = hidden_states.element_size()

    for variant in VARIANTS:
        attention_block = build_attention_block(variant = variant, config = config)
        attention_block.eval()

        with torch.no_grad():
            output, _ = attention_block(hidden_states = hidden_states, attention_mask = None, need_weights = False)

        if baseline_output is None:
            baseline_output = output

        cosine_similarity = torch.nn.functional.cosine_similarity(
            output.reshape(1, -1),
            baseline_output.reshape(1, -1),
            dim = -1
        )
        l2_difference = torch.norm(output - baseline_output)

        kv_channels = variant_kv_channels(
            variant = variant,
            d_model = args.d_model,
            num_heads = args.num_heads,
            num_kv_heads = args.num_kv_heads,
            latent_dim = args.latent_dim
        )
        kv_cache_bytes = estimate_kv_cache_bytes(
            batch_size = args.batch_size,
            seq_len = args.seq_len,
            kv_channels = kv_channels,
            bytes_per_elem = bytes_per_elem
        )

        variant_payloads.append(
            {
                "variant": variant,
                "output_stats": _compute_output_stats(output),
                "cosine_similarity_vs_mha": float(cosine_similarity.item()),
                "l2_difference_vs_mha": float(l2_difference.item()),
                "kv_channels_per_tensor": kv_channels,
                "kv_cache_bytes": kv_cache_bytes,
                "kv_cache_mb": format_bytes_as_megabytes(kv_cache_bytes)
            }
        )

    mha_cache_bytes = next(payload["kv_cache_bytes"] for payload in variant_payloads if payload["variant"] == "mha")
    for payload in variant_payloads:
        payload["compression_ratio_vs_mha"] = float(mha_cache_bytes / max(payload["kv_cache_bytes"], 1))

    kv_image_path = os.path.join(args.image_dir, "kv_cache_comparison.png")
    stats_image_path = os.path.join(args.image_dir, "output_stat_comparison.png")

    _plot_kv_cache_comparison(variant_payloads = variant_payloads, output_path = kv_image_path)
    _plot_output_stats(variant_payloads = variant_payloads, output_path = stats_image_path)

    demo_payload = {
        "meta": {
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "d_model": args.d_model,
            "num_heads": args.num_heads,
            "num_kv_heads": args.num_kv_heads,
            "latent_dim": args.latent_dim,
            "bytes_per_elem": bytes_per_elem
        },
        "variants": variant_payloads,
        "images": {
            "kv_cache_comparison": kv_image_path,
            "output_stat_comparison": stats_image_path
        }
    }

    compare_payload = {}
    if os.path.exists(args.result_path):
        try:
            with open(args.result_path, "r", encoding = "utf-8") as file:
                loaded_payload = json.load(file)
            if isinstance(loaded_payload, dict):
                if "training" in loaded_payload:
                    compare_payload["training"] = loaded_payload["training"]
        except (json.JSONDecodeError, OSError):
            compare_payload = {}

    compare_payload["demo"] = demo_payload

    with open(args.result_path, "w", encoding = "utf-8") as file:
        json.dump(compare_payload, file, ensure_ascii = False, indent = 2)

    logger.info("=" * 80)
    logger.info("Chapter 09 demo completed")
    logger.info(f"Saved comparison json: {args.result_path}")
    logger.info(f"Saved image: {kv_image_path}")
    logger.info(f"Saved image: {stats_image_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers = [logging.StreamHandler()]
    )
    main()
