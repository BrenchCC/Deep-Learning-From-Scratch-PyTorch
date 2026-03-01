"""
Backward-compatible exports for chapter 07 model components.

This file keeps legacy import paths working, while implementations
are split into dedicated modules:
- masks.py
- positional_encoding.py
- attention.py
- feed_forward.py
- encoder.py
- decoder.py
- transformer.py
"""

from .attention import MultiHeadAttention
from .attention import ScaledDotProductAttention
from .decoder import TransformerDecoderLayer
from .encoder import TransformerEncoderLayer
from .feed_forward import PositionwiseFeedForward
from .masks import build_causal_mask
from .masks import build_padding_mask
from .positional_encoding import SinusoidalPositionalEncoding
from .transformer import Seq2SeqTransformer

__all__ = [
    "build_padding_mask",
    "build_causal_mask",
    "SinusoidalPositionalEncoding",
    "ScaledDotProductAttention",
    "MultiHeadAttention",
    "PositionwiseFeedForward",
    "TransformerEncoderLayer",
    "TransformerDecoderLayer",
    "Seq2SeqTransformer"
]
