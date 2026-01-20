import os
import sys
import logging

import torch
import torch.nn as nn

sys.path.append(os.getcwd())

from utils import log_model_info, get_device

logger = logging.getLogger("sentime_lstm_demo") 

class SentimentLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int,
        bidirectional: bool,
        dropout: float,
    ):
        super().__init__()

        # Embedding layer: maps vocabulary indices to dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM layer: processes sequences of embeddings
        self.lstm = nn.LSTM(
            input_size = embedding_dim,
            hidden_size = hidden_dim,
            num_layers = n_layers,
            bidirectional = bidirectional,
            dropout = dropout if n_layers > 1 else 0,
            batch_first = True
        )

        # Fully connected layer: maps LSTM outputs to output dimension
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        """
        text: [batch size, sent len] - LongTensor
        text_lengths: [batch size] - CPU Tensor, packing the sequence lengths
        """
        # [batch size, sent len] -> [batch size, sent len, emb dim]
        embedded = self.dropout(self.embedding(text))

        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded,
            text_lengths,
            batch_first = True,
            enforce_sorted = False
        )

        # packed_outputs is a PackedSequence
        # hidden: (num_layers * num_directions, batch size, hidden dim)
        # cell: (num_layers * num_directions, batch size, hidden dim)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        # Unpack output (如果需要所有时间步的输出，例如 seq2seq)
        # output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        # 获取最后一次的隐状态用于分类
        # hidden 包含 [forward_layer_0, backward_layer_0, forward_layer_1, backward_layer_1...]
        # 我们需要最后两层（最后一层的 fwd 和 bwd）

        hidden_fwd = hidden[-2, :, :]
        hidden_bwd = hidden[-1, :, :]

        # [batch size, hidden dim * 2]
        cat_hidden = torch.cat((hidden_fwd, hidden_bwd), dim = 1)
        return self.fc(self.dropout(cat_hidden))
    
if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler(sys.stdout)]
    )

    device = get_device()
    logger.info(f"Using device: {device}")

    model = SentimentLSTM(
        vocab_size = 1000,
        embedding_dim = 128,
        hidden_dim = 256,
        output_dim = 2,
        n_layers = 2,
        bidirectional = True,
        dropout = 0.5
    ).to(device)

    log_model_info(model)
    logger.info("Model architecture defined specifically for sequence classification.")