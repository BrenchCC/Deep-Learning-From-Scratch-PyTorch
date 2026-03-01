import math
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn

from .decoder import TransformerDecoderLayer
from .encoder import TransformerEncoderLayer
from .masks import build_causal_mask
from .masks import build_padding_mask
from .positional_encoding import SinusoidalPositionalEncoding


class Seq2SeqTransformer(nn.Module):
    """
    Full encoder-decoder Transformer for toy seq2seq tasks.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_heads: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        ffn_hidden_dim: int = 512,
        dropout: float = 0.1,
        max_len: int = 256,
        pad_token_id: int = 0
    ):
        """
        Initialize full Transformer model.

        Args:
            vocab_size (int): Vocabulary size.
            d_model (int): Hidden dimension size.
            num_heads (int): Number of attention heads.
            num_encoder_layers (int): Number of encoder layers.
            num_decoder_layers (int): Number of decoder layers.
            ffn_hidden_dim (int): Feed-forward hidden dimension.
            dropout (float): Dropout rate.
            max_len (int): Maximum sequence length.
            pad_token_id (int): Padding token id.
        """
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_token_id = pad_token_id

        self.src_embedding = nn.Embedding(vocab_size, d_model, padding_idx = pad_token_id)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model, padding_idx = pad_token_id)
        self.positional_encoding = SinusoidalPositionalEncoding(
            d_model = d_model,
            max_len = max_len,
            dropout = dropout
        )

        self.encoder_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model = d_model,
                    num_heads = num_heads,
                    ffn_hidden_dim = ffn_hidden_dim,
                    dropout = dropout
                )
                for _ in range(num_encoder_layers)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    d_model = d_model,
                    num_heads = num_heads,
                    ffn_hidden_dim = ffn_hidden_dim,
                    dropout = dropout
                )
                for _ in range(num_decoder_layers)
            ]
        )

        self.output_projection = nn.Linear(d_model, vocab_size)

    def encode(
        self,
        src_tokens: torch.Tensor,
        src_mask: Optional[torch.Tensor],
        need_weights: bool = False
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Encode source tokens.

        Args:
            src_tokens (torch.Tensor): Source token ids [batch, src_len].
            src_mask (Optional[torch.Tensor]): Source key-padding mask.
            need_weights (bool): Whether to collect attention maps.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: Memory tensor and attention maps.
        """
        x = self.src_embedding(src_tokens) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)

        attention_maps = []
        for layer in self.encoder_layers:
            x, attention_weights = layer(x, src_mask = src_mask, need_weights = need_weights)
            if need_weights and attention_weights is not None:
                attention_maps.append(attention_weights)

        return x, attention_maps

    def decode(
        self,
        tgt_input_tokens: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor],
        memory_mask: Optional[torch.Tensor],
        need_weights: bool = False
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Decode target tokens using encoder memory.

        Args:
            tgt_input_tokens (torch.Tensor): Decoder input ids [batch, tgt_len].
            memory (torch.Tensor): Encoder output [batch, src_len, d_model].
            tgt_mask (Optional[torch.Tensor]): Decoder self-attention mask.
            memory_mask (Optional[torch.Tensor]): Encoder-decoder attention mask.
            need_weights (bool): Whether to collect attention maps.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
                Decoder states, self-attention maps, cross-attention maps.
        """
        x = self.tgt_embedding(tgt_input_tokens) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)

        self_attention_maps = []
        cross_attention_maps = []

        for layer in self.decoder_layers:
            x, self_weights, cross_weights = layer(
                x,
                memory,
                tgt_mask = tgt_mask,
                memory_mask = memory_mask,
                need_weights = need_weights
            )
            if need_weights and self_weights is not None:
                self_attention_maps.append(self_weights)
            if need_weights and cross_weights is not None:
                cross_attention_maps.append(cross_weights)

        return x, self_attention_maps, cross_attention_maps

    def forward(
        self,
        src_tokens: torch.Tensor,
        tgt_input_tokens: torch.Tensor,
        return_attn: bool = False
    ):
        """
        Run end-to-end Transformer forward pass.

        Args:
            src_tokens (torch.Tensor): Source token ids [batch, src_len].
            tgt_input_tokens (torch.Tensor): Decoder input token ids [batch, tgt_len].
            return_attn (bool): Whether to return attention maps.

        Returns:
            torch.Tensor or tuple: Logits or (logits, attention_dict).
        """
        src_mask = build_padding_mask(src_tokens, self.pad_token_id)
        tgt_padding_mask = build_padding_mask(tgt_input_tokens, self.pad_token_id)
        tgt_causal_mask = build_causal_mask(tgt_input_tokens.size(1), tgt_input_tokens.device)
        tgt_mask = tgt_padding_mask | tgt_causal_mask

        memory, encoder_attention_maps = self.encode(
            src_tokens,
            src_mask = src_mask,
            need_weights = return_attn
        )

        decoder_states, decoder_self_maps, decoder_cross_maps = self.decode(
            tgt_input_tokens,
            memory,
            tgt_mask = tgt_mask,
            memory_mask = src_mask,
            need_weights = return_attn
        )
        logits = self.output_projection(decoder_states)

        if return_attn:
            attention_dict: Dict[str, List[torch.Tensor]] = {
                "encoder_self_attn": encoder_attention_maps,
                "decoder_self_attn": decoder_self_maps,
                "decoder_cross_attn": decoder_cross_maps
            }
            return logits, attention_dict
        return logits

    def greedy_decode(
        self,
        src_tokens: torch.Tensor,
        max_len: int,
        bos_token_id: int,
        eos_token_id: int
    ) -> torch.Tensor:
        """
        Generate output tokens with greedy decoding.

        Args:
            src_tokens (torch.Tensor): Source token ids [batch, src_len].
            max_len (int): Maximum generated length excluding BOS.
            bos_token_id (int): Begin-of-sequence token id.
            eos_token_id (int): End-of-sequence token id.

        Returns:
            torch.Tensor: Generated token ids with leading BOS.
        """
        src_mask = build_padding_mask(src_tokens, self.pad_token_id)
        memory, _ = self.encode(src_tokens, src_mask = src_mask, need_weights = False)

        batch_size = src_tokens.size(0)
        generated = torch.full(
            (batch_size, 1),
            bos_token_id,
            dtype = torch.long,
            device = src_tokens.device
        )
        finished = torch.zeros(batch_size, dtype = torch.bool, device = src_tokens.device)

        for _ in range(max_len):
            tgt_padding_mask = build_padding_mask(generated, self.pad_token_id)
            tgt_causal_mask = build_causal_mask(generated.size(1), generated.device)
            tgt_mask = tgt_padding_mask | tgt_causal_mask

            decoder_states, _, _ = self.decode(
                generated,
                memory,
                tgt_mask = tgt_mask,
                memory_mask = src_mask,
                need_weights = False
            )
            next_token_logits = self.output_projection(decoder_states[:, -1:, :])
            next_token = next_token_logits.argmax(dim = -1)

            generated = torch.cat([generated, next_token], dim = 1)
            finished = finished | next_token.squeeze(1).eq(eos_token_id)
            if bool(finished.all()):
                break

        return generated
