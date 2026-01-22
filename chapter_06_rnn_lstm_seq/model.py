import os
import sys
import logging

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

sys.path.append(os.getcwd())

from utils import log_model_info, get_device

logger = logging.getLogger("Model DynamicRNNClassifier")


class DynamicRNNClassifier(nn.Module):
    """
    一个通用的序列分类模型，支持 RNN, LSTM, GRU。
    旨在用于 N-to-1 的分类任务 (Sequence Classification)。
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
        device: torch.device = torch.device('cpu'),
        model_type: str = 'lstm', # Options: 'rnn', 'lstm', 'gru'
    ):
        super().__init__()
        self.device = device
        self.model_type = model_type.lower()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers    
        self.bidirectional = bidirectional
        
        # Embedding Layer
        # padding_idx 会将嵌入向量中对应索引的向量设为全零，避免在计算中使用这些向量。
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)


        # dynamic model selection
        rnn_module_map = {
            'rnn': nn.RNN,
            'lstm': nn.LSTM,
            'gru': nn.GRU,
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
            batch_first = True,
        ).to(self.device)
        
        # output layer
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.fc = nn.Linear(fc_input_dim, output_dim).to(self.device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        """
        Args:
            text: [Batch, Seq_Len] 
            text_lengths: [Batch] 真实长度 (CPU Tensor)
        """
        # [Batch, Seq_Len, Embed_Dim]
        embedded = self.dropout(self.embedding(text))

        # packing
        packed_embedding = pack_padded_sequence(
            embedded,
            text_lengths,
            batch_first = True,
            enforce_sorted = False,
        )

        # ----RNN Pass---
        if self.model_type == 'lstm':
            packed_output, (hidden, cell) = self.rnn(packed_embedding)
        else:
            packed_output, hidden = self.rnn(packed_embedding)

        # ---- Extract Final State ----
        # hidden shape: [n_layers * num_directions, batch, hidden_dim]
        # cell shape: [n_layers * num_directions, batch, hidden_dim]

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
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler(sys.stdout)]
    )
    logger.info("DynamicRNNClassifier Model Test")

    device = get_device("cpu")
    # 测试模型
    vocab_size = 10000
    embedding_dim = 128
    hidden_dim = 256
    output_dim = 2
    num_layers = 2
    bidirectional = True
    dropout = 0.5
    pad_idx = 0

    lstm_model = DynamicRNNClassifier(
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        num_layers,
        bidirectional,
        dropout,
        pad_idx,
        model_type = 'lstm',
    )
    logger.info(f"LSTM Model Info: {log_model_info(lstm_model)}")

    rnn_model = DynamicRNNClassifier(
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        num_layers,
        bidirectional,
        dropout,
        pad_idx,
        model_type = 'rnn',
    )
    logger.info(f"RNN Model Info: {log_model_info(rnn_model)}")
    
    gru_model = DynamicRNNClassifier(
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        num_layers,
        bidirectional,
        dropout,
        pad_idx,
        model_type = 'gru',
    )
    logger.info(f"GRU Model Info: {log_model_info(gru_model)}")

    # 测试前向传播
    input_batch = torch.randint(0, vocab_size, (32, 50))
    input_lengths = torch.randint(10, 50, (32,)).to("cpu")

    logger.info(f"Input batch shape: {input_batch.shape}")
    logger.info(f"Input lengths shape: {input_lengths.shape}")

    # 测试 LSTM 前向传播
    logger.info("Testing LSTM forward pass...")
    lstm_output = lstm_model(input_batch.to(device), input_lengths)
    logger.info(f"LSTM Output shape: {lstm_output.shape}")
    logger.info(f"LSTM Output sample: {lstm_output[0]}")

    # 测试 RNN 前向传播
    logger.info("Testing RNN forward pass...")
    rnn_output = rnn_model(input_batch.to(device), input_lengths)
    logger.info(f"RNN Output shape: {rnn_output.shape}")
    logger.info(f"RNN Output sample: {rnn_output[0]}")

    # 测试 GRU 前向传播
    logger.info("Testing GRU forward pass...")
    gru_output = gru_model(input_batch.to(device), input_lengths)
    logger.info(f"GRU Output shape: {gru_output.shape}")
    logger.info(f"GRU Output sample: {gru_output[0]}")

    logger.info("✅ All model forward passes tested successfully!")

