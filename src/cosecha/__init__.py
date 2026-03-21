"""Cosecha: Tools for harvesting earth observation data for use in flood forecasting."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from cosecha.cosecha import hello
from cosecha.data_models import HarvestedData
from cosecha.logging_config import configure_logging, get_logger
from cosecha.pipeline import HarvestPipeline
from cosecha.reaping import (
    APIError,
    DateRangeError,
    GriddedReaper,
    HRRRReaper,
    InvalidSiteError,
    ReaperError,
    SpatialBoundsError,
    TimeSeriesReaper,
    USGSPrecipReaper,
    USGSStageReaper,
    USGSStreamflowReaper,
)
from cosecha.sowing import (
    DataSower,
    DirectoryError,
    IcebergSower,
    IceChunkSower,
    NetCDFSower,
    ParquetSower,
    SowerError,
    TransformationError,
    WriteError,
    ZarrSower,
)

try:
    __version__ = version("cosecha")
except PackageNotFoundError:
    __version__ = "999"

__all__ = [
    "APIError",
    "DataSower",
    "DateRangeError",
    "DirectoryError",
    "GriddedReaper",
    "HRRRReaper",
    "HarvestPipeline",
    "HarvestedData",
    "IceChunkSower",
    "IcebergSower",
    "InvalidSiteError",
    "NetCDFSower",
    "ParquetSower",
    "ReaperError",
    "SowerError",
    "SpatialBoundsError",
    "TimeSeriesReaper",
    "TransformationError",
    "USGSPrecipReaper",
    "USGSStageReaper",
    "USGSStreamflowReaper",
    "WriteError",
    "ZarrSower",
    "__version__",
    "configure_logging",
    "get_logger",
    "hello",
]
