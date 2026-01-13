import os
import sys
import logging

import torch
import numpy as np
import torch.nn.functional as F

# Insert utils path
sys.path.append(os.getcwd())

try:
    from utils import get_device, setup_seed, Timer, log_model_info
except ImportError:
    # Fallback if utils not present
    def get_device(): return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    def setup_seed(seed): torch.manual_seed(seed); np.random.seed(seed)
    class Timer: 
        def __enter__(self): pass 
        def __exit__(self, *args): pass
    def log_model_info(model): pass

logger = logging.getLogger("ConvDemo")

def conv2d_v1(input_tensor, kernel_weight, bias = 0):
    """
    Perform a naive 2D convolution (cross-correlation) using loops for demonstration.
    Strictly for 1 input channel, 1 output channel, stride=1, no padding.
    """
    # Get shapes
    # input_tensor: (H_in, W_in)
    # kernel_weight: (K, K)
    h_in, w_in = input_tensor.shape
    h_k, w_k = kernel_weight.shape

    # Calculate output shape: H_out = H_in - K + 1
    h_out = h_in - h_k + 1
    w_out = w_in - w_k + 1

    # Initialize output tensor
    output_tensor = torch.zeros((h_out, w_out), dtype = input_tensor.dtype)

    # Silding window convolution
    for i in range(h_out):
        for j in range(w_out):
            # Extract the current window
            # The region of interest (ROI) , which is the region of interest (ROI) in the input tensor
            region = input_tensor[i : i + h_k, j: j + w_k]

            # Compute the dot product
            output_tensor[i, j] = torch.sum(region * kernel_weight) + bias

            # Element-wise multiplication and sum
            val = torch.sum(region * kernel_weight)
            output_tensor[i, j] = val + bias
    
    return output_tensor

def main():
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )
        
    logger.info("Starting Manual Convolution Demo...")

    # Define Data (Simple 5x5 Input, 3x3 Kernel)
    # Using float for calculation
    input_data = torch.tensor([
        [1.0, 1.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 0.0, 0.0]
    ])

    kernel_data = torch.tensor([
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0]
    ])
    
    bias_val = 0.5

    logger.info(f"Input Shape: {input_data.shape}")
    logger.info(f"Kernel Shape: {kernel_data.shape}")

    # Method A: Manual Calculation
    logger.info("Running Manual Calculation (Loop implementation)...")
    manual_output = conv2d_v1(input_data, kernel_data, bias = bias_val)
    
    logger.info("Manual Output Result:")
    logger.info(f"\n{manual_output}")
    
    # Method B: PyTorch Official Implementation
    logger.info("Running PyTorch F.conv2d...")
    
    # Reshape to (Batch, Channel, Height, Width) for PyTorch API
    # Batch = 1, Channel = 1
    input_tensor_pt = input_data.unsqueeze(0).unsqueeze(0)
    kernel_tensor_pt = kernel_data.unsqueeze(0).unsqueeze(0)
    
    torch_output = F.conv2d(input_tensor_pt, kernel_tensor_pt, bias = torch.tensor([bias_val]), stride = 1, padding = 0)
    
    # Squeeze back to 2D for comparison
    torch_output_2d = torch_output.squeeze()
    
    logger.info("PyTorch Output Result:")
    logger.info(f"\n{torch_output_2d}")
    
    # Verification
    diff = torch.abs(manual_output - torch_output_2d).sum().item()
    if diff < 1e-6:
        logger.info("SUCCESS: Manual calculation matches PyTorch implementation exactly!")
    else:
        logger.error(f"FAILURE: Difference detected: {diff}")

if __name__ == "__main__":
    main()
