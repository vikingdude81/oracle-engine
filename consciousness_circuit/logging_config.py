"""
Centralized Logging Configuration for Consciousness Circuit
============================================================

Provides consistent logging across all modules with:
- File + console output
- Configurable log levels
- Structured format with timestamps
- Easy integration

Usage:
    from consciousness_circuit.logging_config import get_logger

    logger = get_logger(__name__)
    logger.info("Starting experiment")
    logger.warning("Low memory")
    logger.error("Failed to load model", exc_info=True)
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


# Global configuration
DEFAULT_LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Track if logging has been configured
_logging_configured = False
_log_file_path: Optional[Path] = None


def setup_logging(
    level: int = DEFAULT_LOG_LEVEL,
    log_file: Optional[str] = None,
    console: bool = True,
) -> Path:
    """
    Configure logging for the entire consciousness_circuit package.

    Args:
        level: Logging level (logging.DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file (default: auto-generated in logs/)
        console: Whether to also log to console

    Returns:
        Path to the log file
    """
    global _logging_configured, _log_file_path

    # Create logs directory
    logs_dir = Path(__file__).parent.parent / "logs"
    logs_dir.mkdir(exist_ok=True)

    # Generate log file name if not provided
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir / f"consciousness_{timestamp}.log"
    else:
        log_file = Path(log_file)

    _log_file_path = log_file

    # Get root logger for consciousness_circuit
    root_logger = logging.getLogger("consciousness_circuit")
    root_logger.setLevel(level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)

    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    _logging_configured = True

    root_logger.info(f"Logging initialized: {log_file}")

    return log_file


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.

    Args:
        name: Module name (typically __name__)

    Returns:
        Configured logger instance

    Example:
        logger = get_logger(__name__)
        logger.info("Processing started")
    """
    global _logging_configured

    # Auto-configure if not done yet
    if not _logging_configured:
        setup_logging()

    # Ensure name is under consciousness_circuit namespace
    if not name.startswith("consciousness_circuit"):
        name = f"consciousness_circuit.{name}"

    return logging.getLogger(name)


def log_exception(logger: logging.Logger, msg: str, exc: Exception) -> None:
    """
    Log an exception with full traceback.

    Args:
        logger: Logger instance
        msg: Context message
        exc: The exception
    """
    logger.error(f"{msg}: {type(exc).__name__}: {exc}", exc_info=True)


def log_gpu_memory(logger: logging.Logger) -> None:
    """Log current GPU memory usage."""
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.debug(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    except ImportError:
        pass


class ExperimentLogger:
    """
    Context manager for experiment logging with timing.

    Usage:
        with ExperimentLogger("layerwise_analysis") as exp:
            exp.log("Starting analysis")
            # ... do work ...
            exp.log("Found peak at layer 42")
    """

    def __init__(self, name: str, logger: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger or get_logger(f"experiment.{name}")
        self.start_time = None

    def __enter__(self):
        import time
        self.start_time = time.time()
        self.logger.info(f"=== Starting: {self.name} ===")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        elapsed = time.time() - self.start_time

        if exc_type is not None:
            self.logger.error(
                f"=== Failed: {self.name} ({elapsed:.1f}s) ===",
                exc_info=(exc_type, exc_val, exc_tb)
            )
            return False  # Re-raise exception
        else:
            self.logger.info(f"=== Completed: {self.name} ({elapsed:.1f}s) ===")
            return True

    def log(self, msg: str, level: int = logging.INFO):
        """Log a message within this experiment context."""
        self.logger.log(level, f"[{self.name}] {msg}")

    def debug(self, msg: str):
        self.log(msg, logging.DEBUG)

    def warning(self, msg: str):
        self.log(msg, logging.WARNING)

    def error(self, msg: str):
        self.log(msg, logging.ERROR)


# Convenience function for quick setup
def quick_setup(verbose: bool = False) -> logging.Logger:
    """
    Quick setup for scripts.

    Args:
        verbose: If True, use DEBUG level; otherwise INFO

    Returns:
        Root logger for consciousness_circuit
    """
    level = logging.DEBUG if verbose else logging.INFO
    setup_logging(level=level)
    return get_logger("consciousness_circuit")
