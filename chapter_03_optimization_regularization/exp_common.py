import os
import sys
import logging

sys.path.append(os.getcwd())

from utils import setup_seed, configure_logging

def setup_experiment(logger_name: str, seed: int = 42):
    """
    Setup logging and random seed for chapter experiments.

    Args:
        logger_name (str): Name used for logger.
        seed (int): Random seed value.

    Returns:
        logging.Logger: Configured logger object.
    """
    configure_logging()
    setup_seed(seed)
    logger = logging.getLogger(logger_name)
    logger.info("=" * 80)
    logger.info(f"{logger_name} initialized with seed = {seed}")
    logger.info("=" * 80)
    return logger

def log_section(logger: logging.Logger, title: str):
    """
    Print a highlighted section header for readable experiment logs.

    Args:
        logger (logging.Logger): Logger instance.
        title (str): Section title.
    """
    logger.info("-" * 60)
    logger.info(title)
    logger.info("-" * 60)

def ensure_parent_dir(path: str):
    """
    Ensure parent directory exists for file path.

    Args:
        path (str): Target file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok = True)
