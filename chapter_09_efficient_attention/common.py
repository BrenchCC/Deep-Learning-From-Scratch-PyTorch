import math
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class AttentionConfig:
    """
    Unified configuration for efficient attention variants.

    Args:
        d_model (int): Hidden size.
        num_heads (int): Number of query heads.
        num_kv_heads (int): Number of key-value heads for GQA.
        latent_dim (int): Latent compression dimension for MLA.
        dropout (float): Dropout rate on attention weights.
    """

    d_model: int = 512
    num_heads: int = 8
    num_kv_heads: int = 2
    latent_dim: int = 64
    dropout: float = 0.1

    def __post_init__(self) -> None:
        """
        Validate configuration values.

        Args:
            None

        Returns:
            None
        """
        if self.d_model <= 0:
            raise ValueError("d_model must be > 0.")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be > 0.")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        if self.num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be > 0.")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads.")
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be > 0.")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("dropout must be in [0, 1).")


def build_causal_mask(seq_len: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Build an upper-triangular causal mask.

    Args:
        seq_len (int): Sequence length.
        device (Optional[torch.device]): Target device.

    Returns:
        torch.Tensor: Boolean mask with shape [1, 1, seq_len, seq_len].
    """
    if seq_len <= 0:
        raise ValueError("seq_len must be > 0.")

    mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype = torch.bool, device = device),
        diagonal = 1
    )
    return mask.unsqueeze(0).unsqueeze(0)


def estimate_kv_cache_bytes(
    batch_size: int,
    seq_len: int,
    kv_channels: int,
    bytes_per_elem: int
) -> int:
    """
    Estimate KV cache memory in bytes.

    Args:
        batch_size (int): Batch size.
        seq_len (int): Sequence length.
        kv_channels (int): Channels per key or per value cache tensor.
        bytes_per_elem (int): Number of bytes per element.

    Returns:
        int: Estimated bytes for both key and value caches.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0.")
    if seq_len <= 0:
        raise ValueError("seq_len must be > 0.")
    if kv_channels <= 0:
        raise ValueError("kv_channels must be > 0.")
    if bytes_per_elem <= 0:
        raise ValueError("bytes_per_elem must be > 0.")

    elements = 2 * batch_size * seq_len * kv_channels
    return int(elements * bytes_per_elem)


def variant_kv_channels(
    variant: str,
    d_model: int,
    num_heads: int,
    num_kv_heads: int,
    latent_dim: int
) -> int:
    """
    Return key/value channel width per cache tensor for one variant.

    Args:
        variant (str): One of mha, mqa, gqa, mla.
        d_model (int): Hidden size.
        num_heads (int): Query head count.
        num_kv_heads (int): Key-value head count for GQA.
        latent_dim (int): Latent dimension for MLA.

    Returns:
        int: Per-tensor cache channels.
    """
    normalized_variant = variant.lower().strip()
    if d_model <= 0 or num_heads <= 0 or num_kv_heads <= 0 or latent_dim <= 0:
        raise ValueError("All dimensions must be > 0.")
    if d_model % num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads.")
    if num_heads % num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads.")

    head_dim = d_model // num_heads
    mapping = {
        "mha": d_model,
        "mqa": head_dim,
        "gqa": num_kv_heads * head_dim,
        "mla": latent_dim
    }

    if normalized_variant not in mapping:
        supported = ", ".join(sorted(mapping.keys()))
        raise ValueError(f"Unsupported variant: {variant}. Supported: {supported}.")

    return int(mapping[normalized_variant])


def format_bytes_as_megabytes(num_bytes: int) -> float:
    """
    Convert bytes to megabytes.

    Args:
        num_bytes (int): Byte count.

    Returns:
        float: Size in MB.
    """
    if num_bytes < 0:
        raise ValueError("num_bytes must be >= 0.")
    return float(num_bytes / (1024.0 ** 2))


def scaled_dot_product(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout: torch.nn.Dropout,
    attention_mask: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute scaled dot-product attention with optional mask.

    Args:
        query (torch.Tensor): Query tensor [B, H, S, D].
        key (torch.Tensor): Key tensor [B, H, S, D].
        value (torch.Tensor): Value tensor [B, H, S, D].
        dropout (torch.nn.Dropout): Dropout layer for attention weights.
        attention_mask (Optional[torch.Tensor]): Boolean mask where True means masked.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Context tensor and attention weights.
    """
    scale = 1.0 / math.sqrt(query.size(-1))
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale

    if attention_mask is not None:
        if attention_mask.dtype != torch.bool:
            raise ValueError("attention_mask must be a boolean tensor.")
        scores = scores.masked_fill(attention_mask, torch.finfo(scores.dtype).min)

    weights = torch.softmax(scores, dim = -1)
    weights = dropout(weights)
    context = torch.matmul(weights, value)
    return context, weights
