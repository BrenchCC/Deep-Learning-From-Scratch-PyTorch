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


class GroupedQueryAttention(nn.Module):
    """
    Grouped-query attention with shared K/V per query group.
    """

    def __init__(self, config: AttentionConfig):
        """
        Initialize GQA module.

        Args:
            config (AttentionConfig): Attention configuration.
        """
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = self.d_model // self.num_heads
        self.group_size = self.num_heads // self.num_kv_heads

        self.query_projection = nn.Linear(self.d_model, self.d_model)
        self.key_projection = nn.Linear(self.d_model, self.num_kv_heads * self.head_dim)
        self.value_projection = nn.Linear(self.d_model, self.num_kv_heads * self.head_dim)
        self.output_projection = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def _split_query_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Split query hidden dimension into query heads.

        Args:
            tensor (torch.Tensor): Query tensor [B, S, D].

        Returns:
            torch.Tensor: Query heads [B, H_q, S, D_h].
        """
        batch_size, seq_len, _ = tensor.shape
        tensor = tensor.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return tensor.transpose(1, 2)

    def _split_kv_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Split hidden dimension into grouped KV heads.

        Args:
            tensor (torch.Tensor): KV tensor [B, S, H_kv * D_h].

        Returns:
            torch.Tensor: KV heads [B, H_kv, S, D_h].
        """
        batch_size, seq_len, _ = tensor.shape
        tensor = tensor.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        return tensor.transpose(1, 2)

    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Merge query heads back to hidden states.

        Args:
            tensor (torch.Tensor): Context [B, H_q, S, D_h].

        Returns:
            torch.Tensor: Hidden states [B, S, D].
        """
        batch_size, _, seq_len, _ = tensor.shape
        tensor = tensor.transpose(1, 2).contiguous()
        return tensor.view(batch_size, seq_len, self.d_model)

    def _expand_grouped_kv(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Expand grouped KV heads to query head count.

        Args:
            tensor (torch.Tensor): Grouped KV [B, H_kv, S, D_h].

        Returns:
            torch.Tensor: Expanded KV [B, H_q, S, D_h].
        """
        return tensor.repeat_interleave(self.group_size, dim = 1)

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
        Run GQA forward pass.

        Args:
            hidden_states (torch.Tensor): Input hidden states [B, S, D].
            attention_mask (Optional[torch.Tensor]): Boolean attention mask.
            need_weights (bool): Whether to return attention weights.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                Output hidden states and optional attention weights.
        """
        normalized_mask = self._normalize_mask(attention_mask)

        query = self._split_query_heads(self.query_projection(hidden_states))
        key = self._split_kv_heads(self.key_projection(hidden_states))
        value = self._split_kv_heads(self.value_projection(hidden_states))

        key = self._expand_grouped_kv(key)
        value = self._expand_grouped_kv(value)

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
