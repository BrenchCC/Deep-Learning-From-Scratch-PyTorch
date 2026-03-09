from typing import Optional, Tuple

import torch
import torch.nn as nn

from chapter_08_transformer_vanilla.attention import MultiHeadAttention
from chapter_08_transformer_vanilla.feed_forward import PositionwiseFeedForward


class TransformerDecoderLayer(nn.Module):
    """
    Single decoder layer.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_hidden_dim: int,
        dropout: float = 0.1
    ):
        """
        Initialize decoder layer.

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
        self.cross_attention = MultiHeadAttention(
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
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Decoder inputs [batch, tgt_len, d_model].
            memory (torch.Tensor): Encoder outputs [batch, src_len, d_model].
            tgt_mask (Optional[torch.Tensor]): Decoder self mask.
            memory_mask (Optional[torch.Tensor]): Cross attention mask.
            need_weights (bool): Whether to return attention maps.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
                Decoder output, self-attn weights, cross-attn weights.
        """
        self_output, self_weights = self.self_attention(
            x,
            x,
            x,
            mask = tgt_mask,
            need_weights = need_weights
        )
        x = self.norm1(x + self.dropout(self_output))

        cross_output, cross_weights = self.cross_attention(
            x,
            memory,
            memory,
            mask = memory_mask,
            need_weights = need_weights
        )
        x = self.norm2(x + self.dropout(cross_output))

        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))

        if need_weights:
            return x, self_weights, cross_weights
        return x, None, None
