import os
import sys
import logging

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

sys.path.append(os.getcwd())

from utils import get_device, log_model_info

logger = logging.getLogger("DynamicRNNClassifier")


class DynamicRNNClassifier(nn.Module):
    """
    Generic sequence classification model supporting RNN, LSTM, and GRU.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.5,
        pad_idx: int = 0,
        device: torch.device = torch.device("cpu"),
        model_type: str = "lstm"
    ):
        """
        Initialize the sequence classifier.

        Args:
            vocab_size (int): Vocabulary size.
            embedding_dim (int): Embedding dimension.
            hidden_dim (int): Hidden size of recurrent layer.
            output_dim (int): Number of output classes.
            num_layers (int): Number of recurrent layers.
            bidirectional (bool): Whether to use bidirectional recurrent layers.
            dropout (float): Dropout probability.
            pad_idx (int): Padding index for embedding.
            device (torch.device): Stored for compatibility with old constructor.
            model_type (str): Recurrent layer type: "rnn", "lstm", or "gru".
        """
        super().__init__()
        self.device = device
        self.model_type = model_type.lower()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)

        rnn_module_map = {
            "rnn": nn.RNN,
            "lstm": nn.LSTM,
            "gru": nn.GRU
        }
        if self.model_type not in rnn_module_map:
            raise ValueError(f"Invalid model_type: {model_type}. Choose from ['rnn', 'lstm', 'gru']")

        rnn_class = rnn_module_map[self.model_type]
        self.rnn = rnn_class(
            input_size = embedding_dim,
            hidden_size = hidden_dim,
            num_layers = num_layers,
            bidirectional = bidirectional,
            dropout = dropout if num_layers > 1 else 0.0,
            batch_first = True
        )

        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        """
        Forward pass for sequence classification.

        Args:
            text (torch.Tensor): Input ids with shape [batch, seq_len].
            text_lengths (torch.Tensor): Real lengths on CPU with shape [batch].

        Returns:
            torch.Tensor: Classification logits.
        """
        embedded = self.dropout(self.embedding(text))

        packed_embedding = pack_padded_sequence(
            embedded,
            text_lengths,
            batch_first = True,
            enforce_sorted = False
        )

        if self.model_type == "lstm":
            _, (hidden, _) = self.rnn(packed_embedding)
        else:
            _, hidden = self.rnn(packed_embedding)

        if self.bidirectional:
            hidden_fwd = hidden[-2, :, :]
            hidden_bwd = hidden[-1, :, :]
            final_hidden = torch.cat((hidden_fwd, hidden_bwd), dim = 1)
        else:
            final_hidden = hidden[-1, :, :]

        return self.fc(self.dropout(final_hidden))


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers = [logging.StreamHandler()]
    )
    logger.info("DynamicRNNClassifier smoke test")

    device = get_device("cpu")
    vocab_size = 10000
    embedding_dim = 128
    hidden_dim = 256
    output_dim = 5

    for model_type in ["lstm", "rnn", "gru"]:
        model = DynamicRNNClassifier(
            vocab_size = vocab_size,
            embedding_dim = embedding_dim,
            hidden_dim = hidden_dim,
            output_dim = output_dim,
            num_layers = 2,
            bidirectional = True,
            dropout = 0.5,
            pad_idx = 0,
            model_type = model_type
        ).to(device)
        log_model_info(model)

        input_batch = torch.randint(0, vocab_size, (8, 20)).to(device)
        input_lengths = torch.randint(5, 20, (8,), dtype = torch.long)
        output = model(input_batch, input_lengths)
        logger.info(f"{model_type.upper()} output shape: {output.shape}")
