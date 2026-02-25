import os
import random
import logging

import torch
import numpy as np

logger = logging.getLogger("Seed_Auto_Config")

def device_optimize(threads: int = 1):
    """
    Configure CPU thread-related environment variables for stable local runs.

    Args:
        threads (int): Number of threads for BLAS/OpenMP backends.
    """
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
    logger.info(f"Device optimize settings initialized: threads = {threads}")

def setup_seed(seed: int = 42):
    """
    Set random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): Random seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    logger.info(f"Python/NumPy seed configured: seed = {seed}")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    logger.info(f"Torch seed configured: seed = {seed}")

def set_seed(seed: int = 42):
    """
    Backward-compatible alias of setup_seed.

    Args:
        seed (int): Random seed value.
    """
    setup_seed(seed = seed)

if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )
    device_optimize(threads = 1)
    setup_seed(seed = 42)
