import logging

def configure_logging(level: int = logging.INFO, clear_handlers: bool = True):
    """
    Configure root logging format for runnable scripts.

    Args:
        level (int): Logging level.
        clear_handlers (bool): Whether to clear existing root handlers first.
    """
    root_logger = logging.getLogger()
    if clear_handlers and root_logger.handlers:
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)

    logging.basicConfig(
        level = level,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )
