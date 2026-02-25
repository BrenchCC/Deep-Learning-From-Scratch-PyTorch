import logging
import time

# Configure logger for this module
logger = logging.getLogger(__name__)

class Timer:
    """
    Context manager to measure execution time of a code block.
    
    Example:
        with Timer("Forward Pass"):
            output = model(input)
    """
    
    def __init__(self, name: str = "Task"):
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.perf_counter()
        logger.info(f"[{self.name}] started...")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.perf_counter() - self.start_time
        if exc_type is not None:
            logger.error(f"[{self.name}] failed with {exc_type.__name__}: {exc_val}")
        logger.info(f"[{self.name}] finished in {elapsed_time:.4f} seconds.")
