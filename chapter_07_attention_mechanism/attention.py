import math
from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention.
    """

    def __init__(self, dropout: float = 0.1):
        """
        Initialize attention module.

        Args:
            dropout (float): Dropout rate on attention weights.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run scaled dot-product attention.

        Args:
            q (torch.Tensor): Query tensor [batch, heads, q_len, d_k].
            k (torch.Tensor): Key tensor [batch, heads, k_len, d_k].
            v (torch.Tensor): Value tensor [batch, heads, k_len, d_k].
            mask (Optional[torch.Tensor]): Attention mask where True means masked.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Context tensor and attention weights.
        """
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            if mask.dtype != torch.bool:
                raise ValueError("Attention mask must be a boolean tensor.")
            mask_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(mask, mask_value)

        attention_weights = torch.softmax(scores, dim = -1)
        attention_weights = self.dropout(attention_weights)
        context = torch.matmul(attention_weights, v)
        return context, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with linear projections.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        Initialize multi-head attention.

        Args:
            d_model (int): Hidden dimension size.
            num_heads (int): Number of heads.
            dropout (float): Dropout rate.
        """
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_projection = nn.Linear(d_model, d_model)
        self.k_projection = nn.Linear(d_model, d_model)
        self.v_projection = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(dropout = dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split hidden dimension into multi-head representation.

        Args:
            x (torch.Tensor): Tensor with shape [batch, seq_len, d_model].

        Returns:
            torch.Tensor: Tensor with shape [batch, heads, seq_len, head_dim].
        """
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Merge multi-head representation back to hidden dimension.

        Args:
            x (torch.Tensor): Tensor with shape [batch, heads, seq_len, head_dim].

        Returns:
            torch.Tensor: Tensor with shape [batch, seq_len, d_model].
        """
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)

    def _normalize_mask(self, mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Normalize mask to 4D attention format.

        Args:
            mask (Optional[torch.Tensor]): Input mask tensor.

        Returns:
            Optional[torch.Tensor]: Mask with shape [batch, 1, q_len, k_len].
        """
        if mask is None:
            return None
        if mask.dim() == 2:
            return mask.unsqueeze(1).unsqueeze(2)
        if mask.dim() == 3:
            return mask.unsqueeze(1)
        if mask.dim() == 4:
            return mask
        raise ValueError("mask must have 2, 3, or 4 dimensions.")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Run multi-head attention.

        Args:
            query (torch.Tensor): Query tensor [batch, q_len, d_model].
            key (torch.Tensor): Key tensor [batch, k_len, d_model].
            value (torch.Tensor): Value tensor [batch, k_len, d_model].
            mask (Optional[torch.Tensor]): Attention mask where True means masked.
            need_weights (bool): Whether to return attention weights.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Output tensor and optional attention map.
        """
        normalized_mask = self._normalize_mask(mask)

        q = self._split_heads(self.q_projection(query))
        k = self._split_heads(self.k_projection(key))
        v = self._split_heads(self.v_projection(value))

        context, attention_weights = self.attention(q, k, v, normalized_mask)
        output = self.output_projection(self._merge_heads(context))

        if need_weights:
            return output, attention_weights
        return output, None
