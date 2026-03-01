from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .feed_forward import PositionwiseFeedForward


class TransformerDecoderLayer(nn.Module):
    """
    Single post-layernorm Transformer decoder layer.
    """

    def __init__(self, d_model: int, num_heads: int, ffn_hidden_dim: int, dropout: float = 0.1):
        """
        Initialize decoder layer.

        Args:
            d_model (int): Hidden dimension size.
            num_heads (int): Number of attention heads.
            ffn_hidden_dim (int): Feed-forward hidden dimension.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model = d_model, num_heads = num_heads, dropout = dropout)
        self.cross_attention = MultiHeadAttention(d_model = d_model, num_heads = num_heads, dropout = dropout)
        self.feed_forward = PositionwiseFeedForward(d_model = d_model, hidden_dim = ffn_hidden_dim, dropout = dropout)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Run one decoder layer.

        Args:
            x (torch.Tensor): Decoder input [batch, tgt_len, d_model].
            memory (torch.Tensor): Encoder output [batch, src_len, d_model].
            tgt_mask (Optional[torch.Tensor]): Decoder self-attention mask.
            memory_mask (Optional[torch.Tensor]): Encoder-decoder attention mask.
            need_weights (bool): Whether to return attention maps.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
                Decoder output, self-attention map, cross-attention map.
        """
        self_attention_output, self_attention_weights = self.self_attention(
            query = x,
            key = x,
            value = x,
            mask = tgt_mask,
            need_weights = need_weights
        )
        x = self.norm_1(x + self.dropout_1(self_attention_output))

        cross_attention_output, cross_attention_weights = self.cross_attention(
            query = x,
            key = memory,
            value = memory,
            mask = memory_mask,
            need_weights = need_weights
        )
        x = self.norm_2(x + self.dropout_2(cross_attention_output))

        ffn_output = self.feed_forward(x)
        x = self.norm_3(x + self.dropout_3(ffn_output))
        return x, self_attention_weights, cross_attention_weights
