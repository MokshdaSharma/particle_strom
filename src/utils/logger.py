import sys
from loguru import logger

def setup_logger(debug_mode: bool = False):
    """Configures the loguru logger."""
    logger.remove()
    
    level = "DEBUG" if debug_mode else "INFO"
    
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True
    )
    
    logger.debug("Logger initialized.")
