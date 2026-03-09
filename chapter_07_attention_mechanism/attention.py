import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention core module.
    """

    def __init__(self, dropout: float = 0.1):
        """
        Initialize attention module.

        Args:
            dropout (float): Dropout rate for attention probabilities.
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
        Compute scaled dot-product attention.

        Args:
            q (torch.Tensor): Query tensor with shape [batch, q_len, d_k].
            k (torch.Tensor): Key tensor with shape [batch, k_len, d_k].
            v (torch.Tensor): Value tensor with shape [batch, k_len, d_v].
            mask (Optional[torch.Tensor]): Boolean mask where True means masked.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                context tensor [batch, q_len, d_v],
                attention scores [batch, q_len, k_len].
        """
        d_k = q.size(-1)
        raw_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            if mask.dtype != torch.bool:
                raise ValueError("mask must be a boolean tensor.")

            if mask.dim() == 4:
                applied_mask = mask.squeeze(1)
            elif mask.dim() == 3:
                applied_mask = mask
            elif mask.dim() == 2:
                applied_mask = mask.unsqueeze(1)
            else:
                raise ValueError("mask must have 2, 3, or 4 dimensions.")

            raw_scores = raw_scores.masked_fill(applied_mask, torch.finfo(raw_scores.dtype).min)

        attention_scores = torch.softmax(raw_scores, dim = -1)
        attention_scores = self.dropout(attention_scores)
        context = torch.matmul(attention_scores, v)
        return context, attention_scores
