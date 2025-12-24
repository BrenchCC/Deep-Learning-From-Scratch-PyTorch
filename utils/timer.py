import time
import logging

# Configure logger for this module
logger = logging.getLogger("Timer")

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
        self.start_time = time.time()
        logger.info(f"[{self.name}] started...")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        logger.info(f"[{self.name}] finished in {elapsed_time:.4f} seconds.")