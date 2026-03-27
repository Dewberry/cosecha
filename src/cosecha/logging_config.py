"""Logging configuration for cosecha.

This module sets up structured logging for the cosecha package using
Python's standard logging module.
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

__all__ = ["configure_logging", "get_logger"]


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the given name.

    Parameters
    ----------
    name : str
        Logger name (typically __name__).

    Returns
    -------
    logging.Logger
        Configured logger instance.

    Examples
    --------
    >>> logger = get_logger(__name__)
    >>> logger.info("Processing data")
    """
    return logging.getLogger(name)


def configure_logging(
    level: int = logging.INFO,
    log_file: Path | str | None = None,
) -> None:
    """Configure logging for cosecha.

    Sets up a root logger for the cosecha package with optional file output.
    Uses a simple format with timestamp, logger name, level, and message.

    Parameters
    ----------
    level : int, optional
        Logging level (logging.DEBUG, logging.INFO, etc.).
        By default logging.INFO.
    log_file : Path | str | None, optional
        Path to log file. If provided, logs will also be written to file.
        By default None (console only).

    Examples
    --------
    >>> import logging
    >>> from pathlib import Path
    >>> configure_logging(level=logging.DEBUG, log_file="cosecha.log")
    """
    logger = logging.getLogger("cosecha")
    logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)

    # Add console handler if not already present
    if not logger.handlers:
        logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
