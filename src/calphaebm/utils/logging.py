# src/calphaebm/utils/logging.py


"""Centralized logging configuration for CalphaEBM.

Provides consistent logging across all modules with clean ASCII output.
"""

import logging
import sys
from typing import Optional


def setup_logger(
    name: str = "calphaebm",
    level: str = "INFO",
    log_file: Optional[str] = None,
    verbose: bool = False,
) -> logging.Logger:
    """Configure and return a logger instance.
    
    Args:
        name: Logger name.
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional file path to write logs.
        verbose: If True, show more details (module, line number).
        
    Returns:
        Configured logger.
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create formatters
    if verbose:
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S"
        )
    
    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# Global logger instance (lazy initialized)
_logger: Optional[logging.Logger] = None


def get_logger() -> logging.Logger:
    """Get the global logger instance."""
    global _logger
    if _logger is None:
        _logger = setup_logger()
    return _logger


def set_log_level(level: str) -> None:
    """Change log level globally."""
    logger = get_logger()
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))


class ProgressBar:
    """Simple ASCII progress bar for loops."""
    
    def __init__(self, total: int, width: int = 50, prefix: str = "Progress"):
        self.total = total
        self.width = width
        self.prefix = prefix
        self.current = 0
        
    def update(self, n: int = 1) -> None:
        """Update progress by n steps."""
        self.current += n
        percent = self.current / self.total
        filled = int(self.width * percent)
        bar = "=" * filled + "-" * (self.width - filled)
        
        sys.stdout.write(f"\r{self.prefix}: [{bar}] {self.current}/{self.total} ({percent:.1%})")
        sys.stdout.flush()
        
        if self.current >= self.total:
            sys.stdout.write("\n")