"""Data models for cosecha harvesting pipeline.

This module defines the core data structures used throughout the cosecha
pipeline to represent harvested data from various sources (NWIS time-series,
NWP gridded data, etc.) with associated metadata.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, Union

import pandas as pd
import xarray as xr

__all__ = [
    "HarvestedData",
    "validate_data",
    "validate_metadata",
    "validate_date_range",
    "validate_spatial_bounds",
]

logger = logging.getLogger(__name__)


def validate_data(data: Union[pd.DataFrame, xr.Dataset]) -> None:
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
        raise ValueError(f"data must be pd.DataFrame or xr.Dataset, got {type(data).__name__}")

    if isinstance(data, pd.DataFrame):
        if data.empty:
            raise ValueError("DataFrame cannot be empty")
    elif isinstance(data, xr.Dataset):
        if not data.data_vars:
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
        raise ValueError("metadata['variable_names'] must be a list or tuple")


def validate_date_range(start_date: str, end_date: str) -> None:
    """Validate that date range is valid (start <= end).

    Parameters
    ----------
    start_date : str
        Start date in ISO 8601 format (YYYY-MM-DD).
    end_date : str
        End date in ISO 8601 format (YYYY-MM-DD).

    Raises
    ------
    ValueError
        If date range is invalid.
    """
    # Strictly validate ISO 8601 format (YYYY-MM-DD)
    import re

    iso_pattern = r"^\d{4}-\d{2}-\d{2}$"

    if not re.match(iso_pattern, str(start_date)):
        raise ValueError(f"Invalid date format: start_date must be YYYY-MM-DD, got {start_date}")
    if not re.match(iso_pattern, str(end_date)):
        raise ValueError(f"Invalid date format: end_date must be YYYY-MM-DD, got {end_date}")

    try:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
    except Exception as e:
        raise ValueError(f"Invalid date format: {e}")

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


@dataclass
class HarvestedData:
    """Container for harvested data with metadata.

    Represents data harvested from a single source (USGS NWIS, HRRR, etc.)
    along with associated metadata. Supports both time-series data (as
    pd.DataFrame) and gridded data (as xr.Dataset).

    Parameters
    ----------
    data : Union[pd.DataFrame, xr.Dataset]
        The harvested data. DataFrames are used for time-series data
        (e.g., NWIS streamflow observations); Datasets are used for
        gridded data (e.g., HRRR forecast grids).
    source_name : str
        Name of the data source (e.g., "USGS_NWIS", "HRRR").
    timestamp : datetime
        When the data was harvested.
    variable_names : list[str]
        Names of variables in the data.
    metadata : dict[str, Any], optional
        Additional metadata (e.g., spatial bounds, units, parameters used).
        By default {}.

    Attributes
    ----------
    data : Union[pd.DataFrame, xr.Dataset]
        The harvested data.

    Raises
    ------
    ValueError
        If data is empty, invalid, or metadata is malformed.

    Examples
    --------
    Time-series data from NWIS:

    >>> import pandas as pd
    >>> from datetime import datetime
    >>> df = pd.DataFrame({
    ...     'time': pd.date_range('2026-01-01', periods=3),
    ...     'site_id': ['01018035', '01018035', '01018035'],
    ...     'value': [1.5, 1.6, 1.7],
    ... })
    >>> data = HarvestedData(
    ...     data=df,
    ...     source_name="USGS_NWIS",
    ...     timestamp=datetime.now(UTC),
    ...     variable_names=["streamflow"],
    ... )

    Gridded data from HRRR:

    >>> import xarray as xr
    >>> import numpy as np
    >>> ds = xr.Dataset({
    ...     'tp': (['init_time', 'latitude', 'longitude'], np.random.rand(1, 10, 10))
    ... })
    >>> data = HarvestedData(
    ...     data=ds,
    ...     source_name="HRRR",
    ...     timestamp=datetime.now(UTC),
    ...     variable_names=["tp"],
    ... )
    """

    data: Union[pd.DataFrame, xr.Dataset]
    source_name: str
    timestamp: datetime
    variable_names: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate harvested data and metadata after initialization."""
        validate_data(self.data)

        # Build metadata dict with required fields
        _metadata = {
            "source_name": self.source_name,
            "timestamp": self.timestamp,
            "variable_names": self.variable_names,
        }
        _metadata.update(self.metadata)

        validate_metadata(_metadata)
        self.metadata = _metadata

        logger.debug(
            f"Created HarvestedData: source={self.source_name}, "
            f"type={self._data_type()}, variables={self.variable_names}"
        )

    def _data_type(self) -> Literal["timeseries", "gridded"]:
        """Return the type of data (timeseries or gridded).

        Returns
        -------
        Literal["timeseries", "gridded"]
            The data type.
        """
        return "timeseries" if isinstance(self.data, pd.DataFrame) else "gridded"

    def is_timeseries(self) -> bool:
        """Check if data is time-series (DataFrame).

        Returns
        -------
        bool
            True if data is a pd.DataFrame.
        """
        return isinstance(self.data, pd.DataFrame)

    def is_gridded(self) -> bool:
        """Check if data is gridded (xarray Dataset).

        Returns
        -------
        bool
            True if data is an xr.Dataset.
        """
        return isinstance(self.data, xr.Dataset)

    def __repr__(self) -> str:
        """Return string representation of HarvestedData."""
        size_info = (
            f"shape={self.data.shape}"
            if isinstance(self.data, pd.DataFrame)
            else f"dims={list(self.data.dims)}"
        )
        return (
            f"HarvestedData(source={self.source_name}, type={self._data_type()}, "
            f"{size_info}, variables={self.variable_names})"
        )
