import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        """
        Initialize positional encoding.

        Args:
            d_model (int): Hidden size.
            max_len (int): Maximum sequence length.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype = torch.float) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.

        Args:
            x (torch.Tensor): Input embeddings [batch, seq_len, d_model].

        Returns:
            torch.Tensor: Position-aware embeddings.
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)
