"""Custom exceptions for the reaping module."""

from __future__ import annotations

__all__ = [
    "ReaperError",
    "InvalidSiteError",
    "DateRangeError",
    "APIError",
    "SpatialBoundsError",
]


class ReaperError(Exception):
    """Base exception for reaper errors."""

    pass


class InvalidSiteError(ReaperError):
    """Raised when an invalid site ID is provided."""

    pass


class DateRangeError(ReaperError):
    """Raised when an invalid date range is provided."""

    pass


class APIError(ReaperError):
    """Raised when an API call fails."""

    pass


class SpatialBoundsError(ReaperError):
    """Raised when invalid spatial bounds are provided."""

    pass
