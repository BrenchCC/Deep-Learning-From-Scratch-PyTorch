import os
import sys

import logging

import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.getcwd())

# Fallback utilities
try:
    from utils import setup_seed
except ImportError:
    def setup_seed(seed): torch.manual_seed(seed)

from chapter_03_optimization_regularization.normalization import BatchNormalization, LayerNormalization, RMSNormalization

logger = logging.getLogger("Exp_Normalization")

def test_norm_stability(norm_layer, inputs, layer_name):
    """
    Helper to run normalization and check stats.
    """
    try:
        norm_layer.train() # Ensure training mode for BN
        output = norm_layer(inputs)
        
        # Check mean and std of output
        # For LN/RMS, we check last dim. For BN, we check batch dim (usually).
        # To simplify, we look at global stats of the output
        mean = output.mean().item()
        std = output.std().item()
        logger.info(f"[{layer_name}] Output Mean: {mean:.4f}, Std: {std:.4f}")
        return output
    except Exception as e:
        logger.error(f"[{layer_name}] Failed: {e}")
        return None

def main():
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()],
        datefmt = '%Y-%m-%d %H:%M:%S'
    )
    
    setup_seed(42)
    
    # Simulation Config
    # Sequence of vectors, simulating NLP tokens
    dim = 64
    seq_len = 10
    
    # Case 1: Standard Batch Size (N=32)
    # Data is not zero-centered, simulating internal covariate shift
    logger.info("--- Case 1: Standard Batch Size (N=32) ---")
    N = 32
    data = torch.randn(N, seq_len, dim) * 5.0 + 10.0 # Shifted Gaussian
    
    # BN (PyTorch API for comparison)
    # BN expects (N, C, L), so we permute (N, dim, seq_len)
    bn_torch = nn.BatchNorm1d(dim)
    data_permuted = data.permute(0, 2, 1)
    test_norm_stability(bn_torch, data_permuted, "PyTorch_BN")

    bn_manual = BatchNormalization(num_features = dim)
    test_norm_stability(bn_manual, data_permuted, "Manual_BN")
    
    # LN (Manual)
    ln_pytorch = nn.LayerNorm(normalized_shape = dim)
    test_norm_stability(ln_pytorch, data, "PyTorch_LN")
    
    ln_manual = LayerNormalization(normalized_shape = dim)
    test_norm_stability(ln_manual, data, "Manual_LN")
    
    # RMS (Manual)
    rms_manual = RMSNormalization(dim = dim)
    test_norm_stability(rms_manual, data, "Manual_RMS")
    
    # Case 2: Extreme Batch Size (N=1) - Inference Style or Large Model Training
    logger.info("-- Case 2: Single Batch (N=1) ---")
    N = 1
    data_single = torch.randn(N, seq_len, dim) * 5.0 + 10.0
    
    # BN (PyTorch API)
    # BN Training with BS=1 will fail to calculate population variance or error out
    bn_torch_single = nn.BatchNorm1d(dim)
    data_single_permuted = data_single.permute(0, 2, 1)
    test_norm_stability(bn_torch_single, data_single_permuted, "PyTorch_BN_BS=1")
    # BN (Manual) - Should also fail or produce invalid output
    bn_manual_single = BatchNormalization(num_features = dim)
    test_norm_stability(bn_manual_single, data_single_permuted, "Manual_BN_BS=1")

    # LN (PyTorch API) - Should result in valid output
    ln_pytorch_single = nn.LayerNorm(normalized_shape = dim)
    test_norm_stability(ln_pytorch_single, data_single, "PyTorch_LN_BS=1")

    # LN (Manual) - Should result in valid output
    test_norm_stability(ln_manual, data_single, "Manual_LN_BS=1")
    
    # RMS (Manual) - Should result in valid output
    test_norm_stability(rms_manual, data_single, "Manual_RMS_BS=1")

    logger.info("Conclusion: LN and RMSNorm are batch-independent, making them crucial for LLMs.")

if __name__ == "__main__":
    main()