import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .attention import ScaledDotProductAttention
from .masks import build_padding_mask


class SingleLayerSelfAttentionModel(nn.Module):
    """
    Single-layer self-attention model for masked copy task.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        pad_token_id: int = 0,
        dropout: float = 0.1
    ):
        """
        Initialize model.

        Args:
            vocab_size (int): Vocabulary size.
            d_model (int): Embedding hidden size.
            pad_token_id (int): Padding token id.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_token_id = pad_token_id

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx = pad_token_id)
        self.q_projection = nn.Linear(d_model, d_model)
        self.k_projection = nn.Linear(d_model, d_model)
        self.v_projection = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout = dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        input_tokens: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_scores: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            input_tokens (torch.Tensor): Input ids [batch, seq_len].
            padding_mask (Optional[torch.Tensor]): Mask [batch, 1, 1, seq_len].
            return_scores (bool): Whether to return attention scores.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Logits and optional scores.
        """
        if padding_mask is None:
            padding_mask = build_padding_mask(input_tokens, self.pad_token_id)

        x = self.embedding(input_tokens) * math.sqrt(self.d_model)
        q = self.q_projection(x)
        k = self.k_projection(x)
        v = self.v_projection(x)

        context, attention_scores = self.attention(q, k, v, mask = padding_mask)
        hidden = self.norm(x + self.dropout(context))
        logits = self.output_projection(hidden)

        if return_scores:
            return logits, attention_scores
        return logits, None
