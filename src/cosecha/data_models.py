"""Data models for cosecha harvesting pipeline.

This module defines the core data structures used throughout the cosecha
pipeline to represent harvested data from various sources (NWIS time-series,
NWP gridded data, etc.) with associated metadata.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
import xarray as xr

__all__ = [
    "validate_data",
    "validate_date_range",
    "validate_metadata",
    "validate_spatial_bounds",
]


def validate_data(data: pd.DataFrame | xr.Dataset) -> None:
    """Validate that data is a DataFrame or xarray Dataset and not empty.

    Parameters
    ----------
    data : Union[pd.DataFrame, xr.Dataset]
        The data to validate.

    Raises
    ------
    ValueError
        If data is not a DataFrame or Dataset, or if data is empty.
    """
    if not isinstance(data, (pd.DataFrame, xr.Dataset)):
        raise TypeError(f"data must be pd.DataFrame or xr.Dataset, got {type(data).__name__}")

    if isinstance(data, pd.DataFrame):
        if data.empty:
            raise ValueError("DataFrame cannot be empty")
    elif isinstance(data, xr.Dataset) and not data.data_vars:
        raise ValueError("xarray Dataset must contain at least one data variable")


def validate_metadata(metadata: dict[str, Any]) -> None:
    """Validate that metadata contains required fields.

    Parameters
    ----------
    metadata : dict[str, Any]
        The metadata dictionary to validate.

    Raises
    ------
    ValueError
        If required fields are missing from metadata.
    """
    required_fields = {"source_name", "timestamp", "variable_names"}
    missing = required_fields - set(metadata.keys())

    if missing:
        raise ValueError(f"metadata missing required fields: {missing}")

    if not isinstance(metadata.get("variable_names"), (list, tuple)):
        raise TypeError("metadata['variable_names'] must be a list or tuple")


def validate_date_range(start_date: str, end_date: str) -> None:
    """Validate that date range is valid (start <= end).

    Parameters
    ----------
    start_date : str
        Start date in ISO 8601 format (e.g. YYYY-MM-DD or YYYY-MM-DDTHH:MMZ).
    end_date : str
        End date in ISO 8601 format (e.g. YYYY-MM-DD or YYYY-MM-DDTHH:MMZ).

    Raises
    ------
    ValueError
        If date range is invalid.
    """
    # Strictly validate ISO 8601 format (YYYY-MM-DD or extended formats with time/timezone)

    iso_pattern = r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?)?$"

    if not re.match(iso_pattern, str(start_date)):
        raise ValueError(
            f"Invalid date format: start_date must be ISO 8601 (e.g., YYYY-MM-DD or YYYY-MM-DDTHH:MMZ), got {start_date}"
        )
    if not re.match(iso_pattern, str(end_date)):
        raise ValueError(
            f"Invalid date format: end_date must be ISO 8601 (e.g., YYYY-MM-DD or YYYY-MM-DDTHH:MMZ), got {end_date}"
        )

    try:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
    except Exception as e:
        raise ValueError(f"Invalid date format: {e}") from e

    if start > end:
        raise ValueError(f"start_date ({start_date}) must be <= end_date ({end_date})")


def validate_spatial_bounds(
    bounds: tuple[float, float, float, float],
) -> None:
    """Validate spatial bounds (lon_min, lat_min, lon_max, lat_max).

    Parameters
    ----------
    bounds : tuple[float, float, float, float]
        Spatial bounds as (lon_min, lat_min, lon_max, lat_max).

    Raises
    ------
    ValueError
        If bounds are invalid (out of range or inverted).
    """
    lon_min, lat_min, lon_max, lat_max = bounds

    if not (-180 <= lon_min <= 180 and -180 <= lon_max <= 180):
        raise ValueError(f"Longitude bounds must be in [-180, 180], got ({lon_min}, {lon_max})")

    if not (-90 <= lat_min <= 90 and -90 <= lat_max <= 90):
        raise ValueError(f"Latitude bounds must be in [-90, 90], got ({lat_min}, {lat_max})")

    if lon_min > lon_max:
        raise ValueError(f"lon_min ({lon_min}) must be <= lon_max ({lon_max})")

    if lat_min > lat_max:
        raise ValueError(f"lat_min ({lat_min}) must be <= lat_max ({lat_max})")
