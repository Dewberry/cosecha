"""Cosecha: Tools for harvesting earth observation data for use in flood forecasting."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from cosecha.logging_config import configure_logging, get_logger
from cosecha.reaping import (
    APIError,
    DateRangeError,
    GriddedReaper,
    InvalidSiteError,
    NWPReaper,
    ReaperError,
    SpatialBoundsError,
    TimeSeriesReaper,
    USGSNWISReaper,
)

try:
    __version__ = version("cosecha")
except PackageNotFoundError:
    __version__ = "999"

__all__ = [
    "APIError",
    "DateRangeError",
    "GriddedReaper",
    "InvalidSiteError",
    "NWPReaper",
    "ReaperError",
    "SpatialBoundsError",
    "TimeSeriesReaper",
    "USGSNWISReaper",
    "__version__",
    "configure_logging",
    "get_logger",
]
