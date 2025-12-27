import sys
import logging

import torch
import torch.nn.functional as F

# Global Logger
logger = logging.getLogger("DimensionOpsDemo")

def analyze_tensor(name, tensor, debug = False):
    """
    Helper to log tensor details clearly.
    
    Args:
        name (str): Description of the tensor.
        tensor (torch.Tensor): The tensor object.
    """
    logger.info(f"[{name}]")
    if debug:
        logger.info(f"  tensor: {tensor}")
    logger.info(f"  Shape: {tensor.shape}")
    logger.info(f"  Contiguous: {tensor.is_contiguous()}")
    logger.info(f"  Strides: {tensor.stride()}")
    logger.info(f"  Device: {tensor.device}")
    logger.info("-" * 30)

def main():
    # Setup logging as preferred: in main, no separate config function
    logging.basicConfig(level = logging.INFO, 
                        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                        handlers = [logging.StreamHandler()])

    logger.info("Starting Transformer-like Dimension Ops Demo...")

    # ==========================================
    # Scenario: Multi-Head Attention Preparation
    # ==========================================
    
    # Hyperparameters
    batch_size = 2
    seq_len = 4
    d_model = 16   # Hidden dimension
    n_heads = 4
    d_head = 4     # d_model / n_heads

    # 1. Input Embedding
    # Shape: (Batch, Seq_Len, D_Model)
    input_embed = torch.randn(batch_size, seq_len, d_model)
    analyze_tensor("Input Embedding", input_embed)

    # 2. Reshape (View) to separate heads
    # We want: (Batch, Seq_Len, N_Heads, D_Head)
    # Note: Using view requires contiguous memory (which it is currently)
    qkv_view = input_embed.view(batch_size, seq_len, n_heads, d_head)
    analyze_tensor("Viewed (Split Heads)", qkv_view)

    # 3. Permute to get heads ready for parallel processing
    # Target: (Batch, N_Heads, Seq_Len, D_Head)
    # This effectively groups processing by head
    qkv_permuted = qkv_view.permute(0, 2, 1, 3)
    analyze_tensor("Permuted (Heads First)", qkv_permuted)

    # ==========================================
    # Scenario: Attention Score Calculation
    # ==========================================

    # 4. Transpose for Dot Product
    # We need Key transposed to (Batch, N_Heads, D_Head, Seq_Len)
    # qkv_permuted acts as both Q and K for this demo
    key_transposed = qkv_permuted.transpose(-2, -1)
    analyze_tensor("Key Transposed", key_transposed)

    # 5. Matrix Multiplication (Not a dim op, but the goal of above ops)
    # (B, H, L, D) @ (B, H, D, L) -> (B, H, L, L)
    attn_scores = torch.matmul(qkv_permuted, key_transposed)
    analyze_tensor("Attention Scores", attn_scores)

    # ==========================================
    # Scenario: Applying Mask (Broadcasting)
    # ==========================================

    # 6. Unsqueeze for Broadcasting
    # Mask shape: (Batch, Seq_Len) -> needs to match (B, H, L, L)
    mask = torch.ones(batch_size, seq_len)
    # Add Head dim and Query_Seq dim
    mask_ready = mask.unsqueeze(1).unsqueeze(2)
    analyze_tensor("Unsqueezed Mask", mask_ready)
    
    # 7. Stack (Simulation of multi-layer output)
    # Suppose we have outputs from 2 different layers
    layer1_out = input_embed
    layer2_out = input_embed
    
    # Stack them to analyze layer variance: (Layers, Batch, Seq, Dim)
    stacked_layers = torch.stack([layer1_out, layer2_out], dim = 0)
    analyze_tensor("Stacked Layers", stacked_layers)

if __name__ == "__main__":
    main()