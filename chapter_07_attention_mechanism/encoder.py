from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .feed_forward import PositionwiseFeedForward


class TransformerEncoderLayer(nn.Module):
    """
    Single post-layernorm Transformer encoder layer.
    """

    def __init__(self, d_model: int, num_heads: int, ffn_hidden_dim: int, dropout: float = 0.1):
        """
        Initialize encoder layer.

        Args:
            d_model (int): Hidden dimension size.
            num_heads (int): Number of attention heads.
            ffn_hidden_dim (int): Feed-forward hidden dimension.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model = d_model, num_heads = num_heads, dropout = dropout)
        self.feed_forward = PositionwiseFeedForward(d_model = d_model, hidden_dim = ffn_hidden_dim, dropout = dropout)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Run one encoder layer.

        Args:
            x (torch.Tensor): Input tensor [batch, src_len, d_model].
            src_mask (Optional[torch.Tensor]): Source key-padding mask.
            need_weights (bool): Whether to return attention map.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Encoded output and optional attention map.
        """
        attention_output, attention_weights = self.self_attention(
            query = x,
            key = x,
            value = x,
            mask = src_mask,
            need_weights = need_weights
        )
        x = self.norm_1(x + self.dropout_1(attention_output))

        ffn_output = self.feed_forward(x)
        x = self.norm_2(x + self.dropout_2(ffn_output))
        return x, attention_weights
