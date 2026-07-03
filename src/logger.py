"""
logger.py
---------
Centralised logging configuration.
All modules call `get_logger(__name__)` to get a named logger.
"""

import logging
import sys
from pathlib import Path

from config import LOG_LEVEL, LOG_FORMAT, LOG_FILE


def setup_logging() -> None:
    """Configure root logger once at app startup."""
    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)

    fmt = logging.Formatter(LOG_FORMAT)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    console.setLevel(level)

    # File handler
    try:
        file_h = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_h.setFormatter(fmt)
        file_h.setLevel(level)
    except OSError:
        file_h = None

    root = logging.getLogger()
    root.setLevel(level)

    if not root.handlers:
        root.addHandler(console)
        if file_h:
            root.addHandler(file_h)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, setting up root logging on first call."""
    setup_logging()
    return logging.getLogger(name)
