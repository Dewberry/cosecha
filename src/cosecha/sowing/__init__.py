"""Data sowing (storage) subpackage for cosecha.

This package provides abstract Protocols and concrete implementations for
writing harvested data to various storage formats (Parquet, Zarr, Iceberg,
IceChunk).
"""

from __future__ import annotations

from cosecha.sowing.base import DataSower
from cosecha.sowing.exceptions import (
    DirectoryError,
    SowerError,
    TransformationError,
    WriteError,
)
from cosecha.sowing.iceberg import IcebergSower
from cosecha.sowing.icechunk import IceChunkSower
from cosecha.sowing.netcdf import NetCDFSower
from cosecha.sowing.parquet import ParquetSower
from cosecha.sowing.zarr import ZarrSower

__all__ = [
    "DataSower",
    "ParquetSower",
    "IcebergSower",
    "ZarrSower",
    "NetCDFSower",
    "IceChunkSower",
    "SowerError",
    "WriteError",
    "DirectoryError",
    "TransformationError",
]
