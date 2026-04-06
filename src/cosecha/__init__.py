"""Cosecha: Tools for harvesting earth observation data for use in flood forecasting."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from cosecha import exceptions
from cosecha.logging import configure_logger
from cosecha.reaping import (
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
    "DateRangeError",
    "GriddedReaper",
    "InvalidSiteError",
    "NWPReaper",
    "ReaperError",
    "SpatialBoundsError",
    "TimeSeriesReaper",
    "USGSNWISReaper",
    "__version__",
    "configure_logger",
    "exceptions",
]
