import torch
import logging

# Configure logger for this module
logger = logging.getLogger("Device Auto Config")

def get_device():
    """
    Select the best available device (CUDA > MPS > CPU).
    Returns:
        torch.device: The selected device object.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Metal Performance Shaders) device for Mac.")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device.")
    
    return device

if __name__ == "__main__":
    # --- Logging Setup ---
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    get_device()