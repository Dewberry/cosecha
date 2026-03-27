"""Custom exceptions for the sowing module."""

from __future__ import annotations

__all__ = [
    "DirectoryError",
    "SowerError",
    "TransformationError",
    "WriteError",
]


class SowerError(Exception):
    """Base exception for sower errors."""


class WriteError(SowerError):
    """Raised when writing data to storage fails."""


class DirectoryError(SowerError):
    """Raised when directory operations fail."""


class TransformationError(SowerError):
    """Raised when data transformations fail."""
