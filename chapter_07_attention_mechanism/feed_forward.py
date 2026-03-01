import torch
import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    """
    Two-layer position-wise feed-forward network.
    """

    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.1):
        """
        Initialize feed-forward module.

        Args:
            d_model (int): Input and output hidden dimension.
            hidden_dim (int): Inner feed-forward hidden dimension.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.linear_1 = nn.Linear(d_model, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run feed-forward transformation.

        Args:
            x (torch.Tensor): Input tensor [batch, seq_len, d_model].

        Returns:
            torch.Tensor: Output tensor [batch, seq_len, d_model].
        """
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.linear_2(x)
