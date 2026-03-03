import math
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn

from .decoder import TransformerDecoderLayer
from .encoder import TransformerEncoderLayer
from .positional_encoding import SinusoidalPositionalEncoding


def build_padding_mask(tokens: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """
    Build padding mask.

    Args:
        tokens (torch.Tensor): Token ids [batch, seq_len].
        pad_token_id (int): Padding token id.

    Returns:
        torch.Tensor: Boolean mask [batch, 1, 1, seq_len].
    """
    return tokens.eq(pad_token_id).unsqueeze(1).unsqueeze(2)


def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Build causal mask.

    Args:
        seq_len (int): Sequence length.
        device (torch.device): Runtime device.

    Returns:
        torch.Tensor: Boolean mask [1, 1, seq_len, seq_len].
    """
    upper_triangle = torch.triu(
        torch.ones(seq_len, seq_len, dtype = torch.bool, device = device),
        diagonal = 1
    )
    return upper_triangle.unsqueeze(0).unsqueeze(1)


class VanillaTransformer(nn.Module):
    """
    Simplified vanilla Transformer encoder-decoder.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_heads: int = 8,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        ffn_hidden_dim: int = 256,
        dropout: float = 0.1,
        max_len: int = 256,
        pad_token_id: int = 0
    ):
        """
        Initialize model.

        Args:
            vocab_size (int): Vocabulary size.
            d_model (int): Hidden size.
            num_heads (int): Number of heads.
            num_encoder_layers (int): Encoder depth.
            num_decoder_layers (int): Decoder depth.
            ffn_hidden_dim (int): FFN hidden size.
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
            src_tokens (torch.Tensor): Source ids [batch, src_len].
            src_mask (Optional[torch.Tensor]): Source key padding mask.
            need_weights (bool): Whether to return attention maps.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: Memory and attention maps.
        """
        x = self.src_embedding(src_tokens) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)

        attention_maps = []
        for layer in self.encoder_layers:
            x, layer_scores = layer(x, src_mask = src_mask, need_weights = need_weights)
            if need_weights and layer_scores is not None:
                attention_maps.append(layer_scores)

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
        Decode target tokens.

        Args:
            tgt_input_tokens (torch.Tensor): Decoder input ids [batch, tgt_len].
            memory (torch.Tensor): Encoder memory [batch, src_len, d_model].
            tgt_mask (Optional[torch.Tensor]): Decoder mask.
            memory_mask (Optional[torch.Tensor]): Cross attention mask.
            need_weights (bool): Whether to return attention maps.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
                Decoder states, self-attention maps, cross-attention maps.
        """
        x = self.tgt_embedding(tgt_input_tokens) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)

        self_maps = []
        cross_maps = []

        for layer in self.decoder_layers:
            x, self_scores, cross_scores = layer(
                x,
                memory,
                tgt_mask = tgt_mask,
                memory_mask = memory_mask,
                need_weights = need_weights
            )
            if need_weights and self_scores is not None:
                self_maps.append(self_scores)
            if need_weights and cross_scores is not None:
                cross_maps.append(cross_scores)

        return x, self_maps, cross_maps

    def forward(
        self,
        src_tokens: torch.Tensor,
        tgt_input_tokens: torch.Tensor,
        return_attn: bool = False
    ):
        """
        End-to-end forward.

        Args:
            src_tokens (torch.Tensor): Source ids.
            tgt_input_tokens (torch.Tensor): Decoder input ids.
            return_attn (bool): Whether to return attention maps.

        Returns:
            torch.Tensor or tuple: logits or (logits, attention_dict).
        """
        src_mask = build_padding_mask(src_tokens, self.pad_token_id)
        tgt_padding_mask = build_padding_mask(tgt_input_tokens, self.pad_token_id)
        tgt_causal_mask = build_causal_mask(tgt_input_tokens.size(1), tgt_input_tokens.device)
        tgt_mask = tgt_padding_mask | tgt_causal_mask

        memory, encoder_maps = self.encode(
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
                "encoder_self_attn": encoder_maps,
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
        Greedy decoding.

        Args:
            src_tokens (torch.Tensor): Source ids [batch, src_len].
            max_len (int): Maximum decode length excluding BOS.
            bos_token_id (int): BOS token id.
            eos_token_id (int): EOS token id.

        Returns:
            torch.Tensor: Generated ids with leading BOS.
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
            next_logits = self.output_projection(decoder_states[:, -1:, :])
            next_token = next_logits.argmax(dim = -1)

            generated = torch.cat([generated, next_token], dim = 1)
            finished = finished | next_token.squeeze(1).eq(eos_token_id)
            if bool(finished.all()):
                break

        return generated
