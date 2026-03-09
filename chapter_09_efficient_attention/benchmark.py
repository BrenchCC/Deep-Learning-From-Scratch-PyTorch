import os
import sys
import json
import time
import logging
import argparse
from typing import Dict, List, Any

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to Python path
sys.path.append(os.getcwd())

from chapter_09_efficient_attention.common import AttentionConfig, format_bytes_as_megabytes, variant_kv_channels, estimate_kv_cache_bytes, build_causal_mask
from chapter_09_efficient_attention.model import build_attention_block
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
    parser = argparse.ArgumentParser(description = "Chapter 09 Efficient Attention Benchmark")
    parser.add_argument("--seed", type = int, default = 42, help = "Random seed")
    parser.add_argument("--batch_size", type = int, default = 1, help = "Batch size")
    parser.add_argument("--d_model", type = int, default = 512, help = "Hidden size")
    parser.add_argument("--num_heads", type = int, default = 8, help = "Number of query heads")
    parser.add_argument("--num_kv_heads", type = int, default = 2, help = "Number of KV heads for GQA")
    parser.add_argument("--latent_dim", type = int, default = 64, help = "Latent dimension for MLA")
    parser.add_argument("--dropout", type = float, default = 0.0, help = "Dropout for benchmark")
    parser.add_argument("--warmup_steps", type = int, default = 10, help = "Warmup steps")
    parser.add_argument("--measure_steps", type = int, default = 30, help = "Measured steps")
    parser.add_argument("--seq_lens", type = str, default = "128,256,512,1024,2048", help = "Comma-separated sequence lengths")
    parser.add_argument(
        "--result_path",
        type = str,
        default = "chapter_09_efficient_attention/results/benchmark_latency_memory.json",
        help = "Output json path"
    )
    parser.add_argument(
        "--image_dir",
        type = str,
        default = "chapter_09_efficient_attention/images",
        help = "Output image directory"
    )
    return parser.parse_args()


def _parse_seq_lens(seq_lens: str) -> List[int]:
    """
    Parse comma-separated sequence lengths.

    Args:
        seq_lens (str): Raw sequence length string.

    Returns:
        List[int]: Parsed integer sequence lengths.
    """
    values = []
    for item in seq_lens.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        values.append(int(stripped))

    if len(values) == 0:
        raise ValueError("seq_lens cannot be empty.")

    return values


def _sync_device(device: torch.device) -> None:
    """
    Synchronize device for reliable timing.

    Args:
        device (torch.device): Runtime device.

    Returns:
        None
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device = device)


def _measure_variant_latency(
    variant: str,
    config: AttentionConfig,
    batch_size: int,
    seq_len: int,
    warmup_steps: int,
    measure_steps: int,
    device: torch.device
) -> float:
    """
    Measure average forward latency for one variant and sequence length.

    Args:
        variant (str): Variant name.
        config (AttentionConfig): Attention config.
        batch_size (int): Batch size.
        seq_len (int): Sequence length.
        warmup_steps (int): Warmup iteration count.
        measure_steps (int): Measured iteration count.
        device (torch.device): Runtime device.

    Returns:
        float: Average latency per forward in seconds.
    """
    module = build_attention_block(variant = variant, config = config).to(device)
    module.eval()

    hidden_states = torch.randn(batch_size, seq_len, config.d_model, device = device)
    attention_mask = build_causal_mask(seq_len = seq_len, device = device)
    attention_mask = attention_mask.expand(batch_size, -1, -1, -1)

    with torch.no_grad():
        for _ in range(warmup_steps):
            _ = module(hidden_states = hidden_states, attention_mask = attention_mask, need_weights = False)

        _sync_device(device)
        start_time = time.perf_counter()
        for _ in range(measure_steps):
            _ = module(hidden_states = hidden_states, attention_mask = attention_mask, need_weights = False)
        _sync_device(device)
        elapsed = time.perf_counter() - start_time

    return float(elapsed / max(measure_steps, 1))


def _plot_latency(entries: List[Dict[str, Any]], output_path: str) -> None:
    """
    Plot latency vs sequence length.

    Args:
        entries (List[Dict[str, Any]]): Benchmark entries.
        output_path (str): Output path.

    Returns:
        None
    """
    plt.figure(figsize = (9, 5))
    for variant in VARIANTS:
        filtered = [item for item in entries if item["variant"] == variant]
        x_axis = [item["seq_len"] for item in filtered]
        y_axis = [item["avg_latency_ms"] for item in filtered]
        plt.plot(x_axis, y_axis, marker = "o", label = variant.upper())

    plt.xlabel("Sequence Length")
    plt.ylabel("Average Latency (ms)")
    plt.title("Chapter 09 Forward Latency Comparison")
    plt.grid(alpha = 0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi = 200)
    plt.close()


def _plot_kv_cache(entries: List[Dict[str, Any]], output_path: str) -> None:
    """
    Plot KV cache size vs sequence length.

    Args:
        entries (List[Dict[str, Any]]): Benchmark entries.
        output_path (str): Output path.

    Returns:
        None
    """
    plt.figure(figsize = (9, 5))
    for variant in VARIANTS:
        filtered = [item for item in entries if item["variant"] == variant]
        x_axis = [item["seq_len"] for item in filtered]
        y_axis = [item["kv_cache_mb"] for item in filtered]
        plt.plot(x_axis, y_axis, marker = "o", label = variant.upper())

    plt.xlabel("Sequence Length")
    plt.ylabel("KV Cache (MB)")
    plt.title("Chapter 09 KV Cache Comparison")
    plt.grid(alpha = 0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi = 200)
    plt.close()


def main() -> None:
    """
    Run latency and memory benchmark across variants.

    Args:
        None

    Returns:
        None
    """
    args = parse_args()
    setup_seed(args.seed)

    os.makedirs(os.path.dirname(args.result_path), exist_ok = True)
    os.makedirs(args.image_dir, exist_ok = True)

    device = get_device()
    seq_lens = _parse_seq_lens(args.seq_lens)

    config = AttentionConfig(
        d_model = args.d_model,
        num_heads = args.num_heads,
        num_kv_heads = args.num_kv_heads,
        latent_dim = args.latent_dim,
        dropout = args.dropout
    )

    entries = []
    bytes_per_elem = torch.tensor(0.0, dtype = torch.float32).element_size()

    loop = tqdm(seq_lens, desc = "SEQ_LEN", leave = False)
    for seq_len in loop:
        for variant in VARIANTS:
            avg_latency = _measure_variant_latency(
                variant = variant,
                config = config,
                batch_size = args.batch_size,
                seq_len = seq_len,
                warmup_steps = args.warmup_steps,
                measure_steps = args.measure_steps,
                device = device
            )

            kv_channels = variant_kv_channels(
                variant = variant,
                d_model = args.d_model,
                num_heads = args.num_heads,
                num_kv_heads = args.num_kv_heads,
                latent_dim = args.latent_dim
            )
            kv_cache_bytes = estimate_kv_cache_bytes(
                batch_size = args.batch_size,
                seq_len = seq_len,
                kv_channels = kv_channels,
                bytes_per_elem = bytes_per_elem
            )

            tokens_per_second = (args.batch_size * seq_len) / max(avg_latency, 1e-12)
            entries.append(
                {
                    "variant": variant,
                    "seq_len": seq_len,
                    "avg_latency_ms": float(avg_latency * 1000.0),
                    "tokens_per_second": float(tokens_per_second),
                    "kv_channels_per_tensor": kv_channels,
                    "kv_cache_bytes": int(kv_cache_bytes),
                    "kv_cache_mb": format_bytes_as_megabytes(kv_cache_bytes)
                }
            )

    latency_plot_path = os.path.join(args.image_dir, "benchmark_latency_curve.png")
    kv_plot_path = os.path.join(args.image_dir, "benchmark_kv_cache_curve.png")

    _plot_latency(entries = entries, output_path = latency_plot_path)
    _plot_kv_cache(entries = entries, output_path = kv_plot_path)

    payload = {
        "meta": {
            "device": str(device),
            "batch_size": args.batch_size,
            "d_model": args.d_model,
            "num_heads": args.num_heads,
            "num_kv_heads": args.num_kv_heads,
            "latent_dim": args.latent_dim,
            "seq_lens": seq_lens,
            "warmup_steps": args.warmup_steps,
            "measure_steps": args.measure_steps,
            "bytes_per_elem": bytes_per_elem
        },
        "entries": entries,
        "images": {
            "latency_curve": latency_plot_path,
            "kv_cache_curve": kv_plot_path
        }
    }

    with open(args.result_path, "w", encoding = "utf-8") as file:
        json.dump(payload, file, ensure_ascii = False, indent = 2)

    logger.info("=" * 80)
    logger.info("Chapter 09 benchmark completed")
    logger.info(f"Saved benchmark json: {args.result_path}")
    logger.info(f"Saved image: {latency_plot_path}")
    logger.info(f"Saved image: {kv_plot_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers = [logging.StreamHandler()]
    )
    main()
