import os
import sys
import logging

import torch
import torch.nn as nn

logger = logging.getLogger("Normalization_Layers")

class BatchNormalization(nn.Module):
    """
    Implementation of Batch Normalization from scratch.
    Based on Paper: Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
    """
    def __init__(self, num_features, eps = 1e-5, momentum = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
        # Running statistics (Not learnable, but part of state_dict)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        # x shape: (N, C) for 1D or (N, C, H, W) for 2D
        # Logic handles (N, C) primarily for MLP contexts
        if self.training:
            # Calculate mean and var along batch dimension (dim 0)
            # For CNN (N, C, H, W), we would reduce over (0, 2, 3)
            if x.dim() == 2:
                mean = x.mean(dim = 0)
                var = x.var(dim = 0, unbiased = False)
            else:
                # Handle simplified 3D case (N, C, L) common in 1D Conv
                mean = x.mean(dim = (0, 2))
                var = x.var(dim = (0, 2), unbiased = False)
            
            # Update running stats
            # running_mean = (1 - m) * running_mean + m * current_mean
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.detach()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.detach()
        else:
            # Use running stats for inference
            mean = self.running_mean
            var = self.running_var

        # Broadcast handling for simple MLP case
        # (x - mean) / sqrt(var + eps) * gamma + beta
        if x.dim() == 2:
            x_norm = (x - mean) / torch.sqrt(var + self.eps)
            return self.gamma * x_norm + self.beta
        elif x.dim() == 3:
            # Reshape for broadcasting over (N, C, L)
            mean = mean.view(1, -1, 1)
            var = var.view(1, -1, 1)
            gamma = self.gamma.view(1, -1, 1)
            beta = self.beta.view(1, -1, 1)
            x_norm = (x - mean) / torch.sqrt(var + self.eps)
            return gamma * x_norm + beta
        return x
    
class LayerNormalization(nn.Module):
    """
    Implementation of Layer Normalization from scratch.
    Based on Paper: Layer Normalization (Ba et al., 2016)
    """
    def __init__(self, normalized_shape, eps = 1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        # Normalize over the last D dimensions defined in normalized_shape
        # Typically the feature dimension
        dims = tuple(range(-len(self.normalized_shape), 0))
        
        mean = x.mean(dim = dims, keepdim = True)
        var = x.var(dim = dims, keepdim = True, unbiased = False)
        
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        return self.gamma * x_norm + self.beta

class RMSNormalization(nn.Module):
    """
    Implementation of RMSNorm.
    Common in Llama / Gemma. 
    Difference: No mean centering, only scaling by RMS.
    """
    def __init__(self, dim, eps = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x shape: (..., Dim)
        input_dtype = x.dtype
        x = x.to(torch.float32)
        
        # RMS = sqrt(mean(x^2))
        variance = x.pow(2).mean(dim = -1, keepdim = True)
        x_norm = x * torch.rsqrt(variance + self.eps)
        
        return self.weight * x_norm.to(input_dtype)

if __name__ == "__main__":
    # Test Block
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()],
        datefmt = '%Y-%m-%d %H:%M:%S'
    )
    
    
    # 1. Test BatchNormalization (Manual vs PyTorch)
    # Input: (Batch=4, Features=8, Seq=2) -> using BatchNorm1d
    x = torch.randn(4, 8, 2)
    
    # Manual
    bn_manual = BatchNormalization(num_features = 8)
    bn_manual.train()
    out_manual = bn_manual(x)
    
    # PyTorch
    bn_torch = nn.BatchNorm1d(num_features = 8, eps = 1e-5, momentum = 0.1)
    # Initialize weights same as manual for fair comparison
    bn_torch.weight.data.fill_(1.0)
    bn_torch.bias.data.fill_(0.0)
    out_torch = bn_torch(x)
    
    diff = (out_manual - out_torch).abs().max()
    logger.info(F"Manual Output Shape: {out_manual}")
    logger.info(F"Torch Output Shape: {out_torch}")

    logger.info(f"Manual  BatchNorm Output Mean (should be close to 0): {out_manual.mean().item():.4f}  ")
    logger.info(f"Manual  BatchNorm Output Std (should be close to 1): {out_manual.std().item():.4f}  ")
    logger.info(f"Torch  BatchNorm Output Mean (should be close to 0): {out_torch.mean().item():.4f}  ")
    logger.info(f"Torch  BatchNorm Output Std (should be close to 1): {out_torch.std().item():.4f}  ")
    logger.info(f"BatchNorm Max Diff (Manual vs Torch): {diff.item():.6f}")

    # 2. Test LayerNormalization
    # Input: (Batch=2, Seq=5, Dim=10)
    x_ln = torch.randn(2, 5, 10)
    
    ln_manual = LayerNormalization(normalized_shape = 10)
    out_ln_manual = ln_manual(x_ln)
    
    ln_torch = nn.LayerNorm(normalized_shape = 10, eps = 1e-5)
    out_ln_torch = ln_torch(x_ln)
    
    diff_ln = (out_ln_manual - out_ln_torch).abs().max()
    logger.info(f"LayerNorm Max Diff (Manual vs Torch): {diff_ln.item():.6f}")

    # 3. Test RMSNormalization
    # PyTorch < 2.4 typically doesn't have nn.RMSNorm, so we just verify functionality
    rms_manual = RMSNormalization(dim = 10)
    out_rms = rms_manual(x_ln)
    logger.info(f"RMSNorm Output Mean (approx 0? No): {out_rms.mean().item():.4f}")
    logger.info(f"RMSNorm Output Std (approx 1? Yes): {out_rms.std().item():.4f}")