import os
import sys
from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn

# Add project root to Python path
sys.path.append(os.getcwd())

from chapter_09_efficient_attention.common import AttentionConfig
from chapter_09_efficient_attention.common import scaled_dot_product


class MultiHeadAttention(nn.Module):
    """
    Standard multi-head attention with independent Q/K/V projections.
    """

    def __init__(self, config: AttentionConfig):
        """
        Initialize MHA module.

        Args:
            config (AttentionConfig): Attention configuration.
        """
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.head_dim = self.d_model // self.num_heads

        self.query_projection = nn.Linear(self.d_model, self.d_model)
        self.key_projection = nn.Linear(self.d_model, self.d_model)
        self.value_projection = nn.Linear(self.d_model, self.d_model)
        self.output_projection = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Split hidden dimension into attention heads.

        Args:
            tensor (torch.Tensor): Input tensor [B, S, D].

        Returns:
            torch.Tensor: Tensor [B, H, S, D_h].
        """
        batch_size, seq_len, _ = tensor.shape
        tensor = tensor.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return tensor.transpose(1, 2)

    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Merge attention heads into hidden dimension.

        Args:
            tensor (torch.Tensor): Input tensor [B, H, S, D_h].

        Returns:
            torch.Tensor: Tensor [B, S, D].
        """
        batch_size, _, seq_len, _ = tensor.shape
        tensor = tensor.transpose(1, 2).contiguous()
        return tensor.view(batch_size, seq_len, self.d_model)

    def _normalize_mask(self, attention_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Normalize mask shape to [B, 1, S_q, S_k] or [B, H, S_q, S_k].

        Args:
            attention_mask (Optional[torch.Tensor]): Input mask.

        Returns:
            Optional[torch.Tensor]: Normalized mask.
        """
        if attention_mask is None:
            return None
        if attention_mask.dim() == 2:
            return attention_mask.unsqueeze(1).unsqueeze(2)
        if attention_mask.dim() == 3:
            return attention_mask.unsqueeze(1)
        if attention_mask.dim() == 4:
            return attention_mask
        raise ValueError("attention_mask must have 2, 3, or 4 dimensions.")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Run MHA forward pass.

        Args:
            hidden_states (torch.Tensor): Input hidden states [B, S, D].
            attention_mask (Optional[torch.Tensor]): Boolean attention mask.
            need_weights (bool): Whether to return attention weights.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                Output hidden states and optional attention weights.
        """
        normalized_mask = self._normalize_mask(attention_mask)

        query = self._split_heads(self.query_projection(hidden_states))
        key = self._split_heads(self.key_projection(hidden_states))
        value = self._split_heads(self.value_projection(hidden_states))

        context, attention_weights = scaled_dot_product(
            query = query,
            key = key,
            value = value,
            dropout = self.dropout,
            attention_mask = normalized_mask
        )
        merged_context = self._merge_heads(context)
        output = self.output_projection(merged_context)

        if need_weights:
            return output, attention_weights
        return output, None
