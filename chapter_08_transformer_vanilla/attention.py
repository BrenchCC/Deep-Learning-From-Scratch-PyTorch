import math
from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention module.
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
            q (torch.Tensor): Query [batch, heads, q_len, d_k].
            k (torch.Tensor): Key [batch, heads, k_len, d_k].
            v (torch.Tensor): Value [batch, heads, k_len, d_k].
            mask (Optional[torch.Tensor]): Boolean mask, True means masked.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Context and attention scores.
        """
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            if mask.dtype != torch.bool:
                raise ValueError("mask must be a boolean tensor.")
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

        attention_scores = torch.softmax(scores, dim = -1)
        attention_scores = self.dropout(attention_scores)
        context = torch.matmul(attention_scores, v)
        return context, attention_scores


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        Initialize module.

        Args:
            d_model (int): Hidden size.
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
        Split hidden dimension into heads.

        Args:
            x (torch.Tensor): Input [batch, seq_len, d_model].

        Returns:
            torch.Tensor: Output [batch, heads, seq_len, head_dim].
        """
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Merge heads to hidden dimension.

        Args:
            x (torch.Tensor): Input [batch, heads, seq_len, head_dim].

        Returns:
            torch.Tensor: Output [batch, seq_len, d_model].
        """
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)

    def _normalize_mask(self, mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Normalize mask shape to [batch, 1, q_len, k_len].

        Args:
            mask (Optional[torch.Tensor]): Input mask.

        Returns:
            Optional[torch.Tensor]: Normalized mask.
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
        Forward pass.

        Args:
            query (torch.Tensor): Query [batch, q_len, d_model].
            key (torch.Tensor): Key [batch, k_len, d_model].
            value (torch.Tensor): Value [batch, k_len, d_model].
            mask (Optional[torch.Tensor]): Attention mask.
            need_weights (bool): Whether to return attention maps.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Output and optional attention maps.
        """
        normalized_mask = self._normalize_mask(mask)

        q = self._split_heads(self.q_projection(query))
        k = self._split_heads(self.k_projection(key))
        v = self._split_heads(self.v_projection(value))

        context, attention_scores = self.attention(q, k, v, mask = normalized_mask)
        output = self.output_projection(self._merge_heads(context))

        if need_weights:
            return output, attention_scores
        return output, None
