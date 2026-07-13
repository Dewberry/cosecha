"""Custom exceptions for the reaping module."""

from __future__ import annotations

__all__ = [
    "APIError",
    "DataNotFoundError",
    "DateRangeError",
    "InvalidSiteError",
    "ReaperError",
    "TransformationError",
]


class ReaperError(Exception):
    """Base exception for reaper errors."""


class InvalidSiteError(ReaperError):
    """Raised when an invalid site ID is provided."""


class DateRangeError(ReaperError):
    """Raised when an invalid date range is provided."""


class APIError(ReaperError):
    """Raised when an API call fails."""


class DataNotFoundError(ReaperError):
    """Raised when a query returns no data."""


class TransformationError(ReaperError):
    """Raised when data transformation fails."""
