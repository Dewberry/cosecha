"""Data reaping (harvesting) subpackage for cosecha.

This package provides abstract base classes and concrete implementations for
fetching data from various sources (USGS NWIS, NWP models, MRMS, etc.).
"""

from __future__ import annotations

from cosecha.reaping.base import GriddedReaper, TimeSeriesReaper
from cosecha.reaping.mrms import MRMSReaper
from cosecha.reaping.nwis import USGSNWISReaper
from cosecha.reaping.nwp import NWPReaper
from cosecha.reaping.usace import ReservoirReaper

__all__ = [
    "GriddedReaper",
    "MRMSReaper",
    "NWPReaper",
    "ReservoirReaper",
    "TimeSeriesReaper",
    "USGSNWISReaper",
]
