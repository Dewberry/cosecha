"""Custom exceptions for the sowing module."""

from __future__ import annotations

__all__ = [
    "SowerError",
    "WriteError",
    "DirectoryError",
    "TransformationError",
]


class SowerError(Exception):
    """Base exception for sower errors."""

    pass


class WriteError(SowerError):
    """Raised when writing data to storage fails."""

    pass


class DirectoryError(SowerError):
    """Raised when directory operations fail."""

    pass


class TransformationError(SowerError):
    """Raised when data transformations fail."""

    pass
