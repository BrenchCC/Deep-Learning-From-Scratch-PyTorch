import os
import random 
import logging

import torch
import numpy as np

logger = logging.getLogger("Seed Auto Config")
def device_optimize():
    """
    Optimize device settings for torch backend using mps.
    """
    # for mac device
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    logger.info("Device optimize settings initialized successfully for torch backend using mps.")

def setup_seed(seed = 42):
    """
    Set random seed for reproducibility across numpy, torch, and python.
    """
    # basic config
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    logger.info("Basic seed settings initialized successfully.")

    # torch config
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    logger.info("Torch seed settings initialized successfully.")

if __name__ == "__main__":
    # --- Logging Setup ---
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    device_optimize()
    setup_seed()
