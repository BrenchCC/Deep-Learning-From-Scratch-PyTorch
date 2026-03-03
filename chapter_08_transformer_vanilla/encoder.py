from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .feed_forward import PositionwiseFeedForward


class TransformerEncoderLayer(nn.Module):
    """
    Single encoder layer.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_hidden_dim: int,
        dropout: float = 0.1
    ):
        """
        Initialize encoder layer.

        Args:
            d_model (int): Hidden size.
            num_heads (int): Number of attention heads.
            ffn_hidden_dim (int): FFN hidden size.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.self_attention = MultiHeadAttention(
            d_model = d_model,
            num_heads = num_heads,
            dropout = dropout
        )
        self.ffn = PositionwiseFeedForward(
            d_model = d_model,
            hidden_dim = ffn_hidden_dim,
            dropout = dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input [batch, src_len, d_model].
            src_mask (Optional[torch.Tensor]): Source key padding mask.
            need_weights (bool): Whether to return attention weights.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Output and optional weights.
        """
        attn_output, attn_weights = self.self_attention(
            x,
            x,
            x,
            mask = src_mask,
            need_weights = need_weights
        )
        x = self.norm1(x + self.dropout(attn_output))

        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))

        if need_weights:
            return x, attn_weights
        return x, None
