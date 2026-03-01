import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding module.
    """

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        """
        Initialize positional encoding table.

        Args:
            d_model (int): Hidden dimension size.
            max_len (int): Maximum supported sequence length.
            dropout (float): Dropout rate after adding position encoding.
        """
        super().__init__()

        position = torch.arange(0, max_len, dtype = torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype = torch.float32) *
            (-math.log(10000.0) / d_model)
        )

        positional_table = torch.zeros(max_len, d_model, dtype = torch.float32)
        positional_table[:, 0::2] = torch.sin(position * div_term)
        positional_table[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("positional_table", positional_table.unsqueeze(0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional information to token embeddings.

        Args:
            x (torch.Tensor): Token embeddings with shape [batch, seq_len, d_model].

        Returns:
            torch.Tensor: Encoded embeddings with same shape as input.
        """
        seq_len = x.size(1)
        x = x + self.positional_table[:, :seq_len, :]
        return self.dropout(x)
