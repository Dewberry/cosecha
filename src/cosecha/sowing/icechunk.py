"""IceChunkSower for writing gridded data to IceChunk format.

This module implements IceChunk-based data writing for gridded data.
IceChunk provides versioning and time-travel capabilities for cloud-native
chunked array storage, similar to Zarr but with transaction support.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

import icechunk

from cosecha.sowing.exceptions import SowerError, WriteError
from cosecha.logging_config import get_logger

if TYPE_CHECKING:
    from cosecha.data_models import HarvestedData
    from cosecha.sowing.base import DataSower

__all__ = ["IceChunkSower"]

logger = get_logger(__name__)


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

        storage = icechunk.local_filesystem_storage(str(self.storage_path))
        try:
            self.repo = icechunk.Repository.open(storage)
            logger.debug("Opened existing IceChunk repository")
        except Exception:
            self.repo = icechunk.Repository.create(storage)
            logger.debug("Created new IceChunk repository")

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

    def sow(self, data: HarvestedData) -> str:
        """Write HarvestedData to IceChunk store.

        Parameters
        ----------
        data : HarvestedData
            The harvested data to write.

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
        >>> path = sower.sow(harvested)
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

            source_clean = re.sub(r"[^a-z0-9_]", "_", data.source_name.lower()).strip("_")
            timestamp_str = data.timestamp.strftime("%Y%m%d_%H%M%S")
            group_path = f"{source_clean}/{timestamp_str}"

            session = self.repo.writable_session("main")

            # IceChunk exclusively uses Zarr v3 specification
            ds.to_zarr(
                store=session.store, group=group_path, mode="w", zarr_format=3, consolidated=False
            )

            commit_msg = f"Appended {data.source_name} data for {timestamp_str}"
            session.commit(commit_msg)

            logger.info(f"Successfully committed to IceChunk repo at group: {group_path}")
            return str(self.storage_path / group_path)

        except SowerError:
            raise
        except Exception as e:
            logger.exception("Failed to write to IceChunk")
            raise WriteError(f"IceChunk write failed: {e}") from e


# Type hint: IceChunkSower implements DataSower protocol
_: type[DataSower] = IceChunkSower
