import torch
import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    """

    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.1):
        """
        Initialize FFN.

        Args:
            d_model (int): Hidden size.
            hidden_dim (int): FFN hidden size.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run FFN.

        Args:
            x (torch.Tensor): Input tensor [batch, seq_len, d_model].

        Returns:
            torch.Tensor: Output tensor [batch, seq_len, d_model].
        """
        return self.net(x)
