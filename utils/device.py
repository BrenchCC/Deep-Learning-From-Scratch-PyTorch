import logging

import torch

# Configure logger for this module
logger = logging.getLogger("Device_Auto_Config")

def get_device(mode: str = 'auto'):
    """
    Select the best available device (CUDA > MPS > CPU).
    Returns:
        torch.device: The selected device object.
    """
    valid_modes = ['auto', 'cuda', 'mps', 'cpu']
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode: {mode}. Choose from {valid_modes}")

    if mode == 'auto':
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using MPS (Metal Performance Shaders) device for Mac.")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU device.")
    elif mode == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA mode requested but CUDA is not available.")
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif mode == 'mps':
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS mode requested but MPS is not available.")
        device = torch.device("mps")
        logger.info("Using MPS (Metal Performance Shaders) device for Mac.")
    elif mode == 'cpu':
        device = torch.device("cpu")
        logger.info("Using CPU device.")
    
    return device

if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )
    
    get_device()
