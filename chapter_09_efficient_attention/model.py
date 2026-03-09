import os
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn

# Add project root to Python path
sys.path.append(os.getcwd())

from chapter_09_efficient_attention.gqa import GroupedQueryAttention
from chapter_09_efficient_attention.mha import MultiHeadAttention
from chapter_09_efficient_attention.mla import MultiHeadLatentAttention
from chapter_09_efficient_attention.mqa import MultiQueryAttention
from chapter_09_efficient_attention.common import AttentionConfig, build_causal_mask


def build_attention_block(variant: str, config: AttentionConfig) -> nn.Module:
    """
    Build one attention block by variant name.

    Args:
        variant (str): One of mha, mqa, gqa, mla.
        config (AttentionConfig): Attention config.

    Returns:
        nn.Module: Attention module instance.
    """
    normalized_variant = variant.lower().strip()
    mapping = {
        "mha": MultiHeadAttention,
        "mqa": MultiQueryAttention,
        "gqa": GroupedQueryAttention,
        "mla": MultiHeadLatentAttention
    }

    if normalized_variant not in mapping:
        supported = ", ".join(sorted(mapping.keys()))
        raise ValueError(f"Unsupported variant: {variant}. Supported: {supported}.")

    return mapping[normalized_variant](config)


class EfficientAttentionLM(nn.Module):
    """
    Minimal language model shell with pluggable attention variant.
    """

    def __init__(
        self,
        vocab_size: int,
        attention_variant: str,
        config: AttentionConfig,
        pad_token_id: int
    ):
        """
        Initialize model.

        Args:
            vocab_size (int): Vocabulary size.
            attention_variant (str): Attention variant name.
            config (AttentionConfig): Attention config.
            pad_token_id (int): Padding token id.
        """
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be > 0.")

        self.vocab_size = vocab_size
        self.attention_variant = attention_variant.lower().strip()
        self.config = config
        self.pad_token_id = pad_token_id

        self.embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim = config.d_model)
        self.attention_block = build_attention_block(self.attention_variant, config)
        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, vocab_size)

    def _build_attention_mask(
        self,
        input_ids: torch.Tensor,
        padding_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Build joint causal + padding attention mask.

        Args:
            input_ids (torch.Tensor): Input ids [B, S].
            padding_mask (Optional[torch.Tensor]): Padding mask [B, S], True means pad.

        Returns:
            torch.Tensor: Boolean mask [B, 1, S, S].
        """
        batch_size, seq_len = input_ids.shape
        causal_mask = build_causal_mask(seq_len = seq_len, device = input_ids.device)
        causal_mask = causal_mask.expand(batch_size, -1, -1, -1)

        if padding_mask is None:
            return causal_mask

        if padding_mask.dtype != torch.bool:
            raise ValueError("padding_mask must be a boolean tensor.")

        key_padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
        return causal_mask | key_padding_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Run model forward pass.

        Args:
            input_ids (torch.Tensor): Input ids [B, S].
            padding_mask (Optional[torch.Tensor]): Padding mask [B, S].
            need_weights (bool): Whether to return attention weights.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                Logits [B, S, V] and optional attention weights.
        """
        hidden_states = self.embedding(input_ids)
        attention_mask = self._build_attention_mask(input_ids, padding_mask)

        attended_states, attention_weights = self.attention_block(
            hidden_states = hidden_states,
            attention_mask = attention_mask,
            need_weights = need_weights
        )

        normalized_states = self.norm(attended_states)
        logits = self.lm_head(normalized_states)

        if need_weights:
            return logits, attention_weights
        return logits, None
