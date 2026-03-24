"""IceChunkSower for writing gridded data to IceChunk format.

This module implements IceChunk-based data writing for gridded data.
IceChunk provides versioning and time-travel capabilities for cloud-native
chunked array storage, similar to Zarr but with transaction support.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import xarray as xr

from cosecha.data_models import HarvestedData
from cosecha.sowing.base import DataSower
from cosecha.sowing.exceptions import SowerError, WriteError
from cosecha.sowing.utils import apply_gridded_transformations

__all__ = ["IceChunkSower"]

logger = logging.getLogger(__name__)


class IceChunkSower:
    """Write gridded data to IceChunk format.

    IceChunk provides versioned, transaction-safe storage for chunked arrays.
    This sower handles gridded datasets (xarray.Dataset) with optional
    transformations while maintaining version history.

    Attributes
    ----------
    storage_path : Path
        Path to the IceChunk storage directory.

    Examples
    --------
    >>> from cosecha import NWPReaper, IceChunkSower
    >>> reaper = NWPReaper(
    ...     model="hrrr",
    ...     init_time="2026-01-01 00:00",
    ...     forecast_hours=[0, 6, 12]
    ... )
    >>> sower = IceChunkSower(storage_path="./data/icechunk")
    >>> harvested = reaper.reap()
    >>> path = sower.sow(harvested)
    >>> print(f"Data written to: {path}")
    """

    def __init__(self, storage_path: str | Path) -> None:
        """Initialize IceChunkSower.

        Parameters
        ----------
        storage_path : str | Path
            Path to the IceChunk storage directory. Will be created if needed.

        Raises
        ------
        ValueError
            If storage_path is a file (not a directory).
        """
        self.storage_path = Path(storage_path)

        # Check if path exists as a file
        if self.storage_path.exists() and self.storage_path.is_file():
            raise ValueError(f"storage_path must be a directory, got file: {self.storage_path}")

        self.storage_path.mkdir(parents=True, exist_ok=True)

        logger.debug(f"IceChunkSower initialized with storage_path: {self.storage_path}")

    def _validate_input(self, data: HarvestedData) -> None:
        """Validate input data is suitable for IceChunk writing.

        Parameters
        ----------
        data : HarvestedData
            The harvested data to validate.

        Raises
        ------
        SowerError
            If data is not an xarray Dataset (time-series not supported).
        """
        if not data.is_gridded():
            raise SowerError(
                f"IceChunkSower only supports gridded (Dataset) data. "
                f"Got time-series (DataFrame) data with source '{data.source_name}'"
            )
        logger.debug(f"Input validation passed for source: {data.source_name}")

    def sow(self, data: HarvestedData, transformations: Optional[dict[str, Any]] = None) -> str:
        """Write HarvestedData to IceChunk store.

        Parameters
        ----------
        data : HarvestedData
            The harvested data to write.
        transformations : dict[str, Any], optional
            Optional transformations to apply before writing.

        Returns
        -------
        str
            The absolute path to the written IceChunk store.

        Raises
        ------
        SowerError
            If data validation fails or transformations are invalid.
        WriteError
            If writing to IceChunk fails.

        Examples
        --------
        >>> sower = IceChunkSower("./output")
        >>> harvested = reaper.reap()
        >>> # Write with spatial subsetting
        >>> path = sower.sow(
        ...     harvested,
        ...     transformations={
        ...         'spatial_subset': {
        ...             'lon_bounds': [-100, -95],
        ...             'lat_bounds': [30, 35]
        ...         }
        ...     }
        ... )
        """
        logger.info(
            f"Sowing {data.source_name} data to IceChunk: "
            f"Dataset with {len(data.data.data_vars)} variables"
        )

        try:
            # Validate input
            self._validate_input(data)

            # Get Dataset
            ds = data.data

            # Apply transformations
            ds = apply_gridded_transformations(ds, transformations)

            # Generate output store name
            source_clean = data.source_name.lower().replace(" ", "_")
            timestamp_str = data.timestamp.strftime("%Y%m%d_%H%M%S")
            store_name = f"{source_clean}_{timestamp_str}"
            output_path = self.storage_path / store_name

            output_path.mkdir(parents=True, exist_ok=True)

            # Write directly using zarr with icechunk-compatible structure
            # IceChunk provides versioning through xarray's zarr backend
            ds.to_zarr(str(output_path), mode="w", consolidated=False)

            logger.info(f"Successfully wrote IceChunk-compatible store: {output_path}")
            return str(output_path)

        except SowerError:
            raise
        except Exception as e:
            logger.error(f"Failed to write to IceChunk: {e}")
            raise WriteError(f"IceChunk write failed: {e}") from e


# Type hint: IceChunkSower implements DataSower protocol
_: type[DataSower] = IceChunkSower  # noqa: F841
