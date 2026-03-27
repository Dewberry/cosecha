"""Data reaping (harvesting) subpackage for cosecha.

This package provides abstract Protocols and concrete implementations for
fetching data from various sources (USGS NWIS, HRRR, etc.).
"""

from __future__ import annotations

from cosecha.reaping.base import GriddedReaper, TimeSeriesReaper
from cosecha.reaping.exceptions import (
    APIError,
    DateRangeError,
    InvalidSiteError,
    ReaperError,
    SpatialBoundsError,
)
from cosecha.reaping.nwis import USGSPrecipReaper, USGSStageReaper, USGSStreamflowReaper
from cosecha.reaping.nwp import NWPReaper

__all__ = [
    "APIError",
    "DateRangeError",
    "GriddedReaper",
    "InvalidSiteError",
    "NWPReaper",
    "ReaperError",
    "SpatialBoundsError",
    "TimeSeriesReaper",
    "USGSPrecipReaper",
    "USGSStageReaper",
    "USGSStreamflowReaper",
]
