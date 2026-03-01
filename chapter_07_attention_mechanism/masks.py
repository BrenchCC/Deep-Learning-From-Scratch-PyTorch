import torch


def build_padding_mask(tokens: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """
    Build key-padding mask where padded positions are masked.

    Args:
        tokens (torch.Tensor): Token ids with shape [batch, seq_len].
        pad_token_id (int): Padding token id.

    Returns:
        torch.Tensor: Boolean mask with shape [batch, 1, 1, seq_len].
    """
    return tokens.eq(pad_token_id).unsqueeze(1).unsqueeze(2)


def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Build upper-triangular future mask for autoregressive decoding.

    Args:
        seq_len (int): Sequence length.
        device (torch.device): Runtime device.

    Returns:
        torch.Tensor: Boolean mask with shape [1, 1, seq_len, seq_len].
    """
    future_mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype = torch.bool, device = device),
        diagonal = 1
    )
    return future_mask.unsqueeze(0).unsqueeze(1)
