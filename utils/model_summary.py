import logging
from typing import Optional

import torch
import torch.nn as nn

# Configure logger for this module
logger = logging.getLogger("Model_Summary_Tool")


def count_parameters(model: nn.Module, only_trainable: bool = True):
    """
    Count the number of parameters in a PyTorch model.
    
    Args:
        model: The PyTorch model.
        only_trainable: If True, only count parameters capable of being updated.
        
    Returns:
        int: The number of parameters.
    """
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

def estimate_model_size(model: nn.Module):
    """
    Estimate the model size in Megabytes (MB) based on parameter count.
    Assuming float32 (4 bytes per parameter).
    
    Args:
        model: The PyTorch model.
        
    Returns:
        float: Estimated size in MB.
    """
    param_count = count_parameters(model, only_trainable = False)
    size_mb = (param_count * 4) / (1024 ** 2)
    return size_mb

def log_model_info(model: nn.Module):
    """
    Log detailed model information including architecture and parameter count.
    
    Args:
        model: The PyTorch model.
    """
    total_params = count_parameters(model, only_trainable = False)
    trainable_params = count_parameters(model, only_trainable = True)
    size_mb = estimate_model_size(model)
    
    logger.info(f"Model Architecture:\n{model}")
    logger.info(f"Total Parameters: {total_params:,}")
    logger.info(f"Trainable Parameters: {trainable_params:,}")
    logger.info(f"Estimated Model Size (fp32): {size_mb:.2f} MB")

def log_model_info_from_path(path: str, model: Optional[nn.Module] = None):
    """
    Load the model state dictionary from a file.
    
    Args:
        path: Path to the saved model state dictionary.
        model: The PyTorch model instance.
    """
    if model is None:
        raise ValueError("model must be provided when loading state dict from path.")

    model.load_state_dict(torch.load(path, map_location = "cpu"))
    logger.info(f"Model loaded from {path}.")
    log_model_info(model)
    
if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )

    model = nn.Linear(10, 2)
    log_model_info(model)
    
