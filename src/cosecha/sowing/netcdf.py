"""NetCDFSower for writing gridded data to NetCDF format.

This module implements NetCDF-based data writing for gridded data
(e.g., HRRR NWP forecasts). NetCDF is a widely-used binary format for
scientific data storage with excellent compression and metadata support.
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

__all__ = ["NetCDFSower"]

logger = logging.getLogger(__name__)


class NetCDFSower:
    """Write gridded data to NetCDF format.

    NetCDF is a self-describing binary format widely used in earth sciences
    for storing scientific data. This sower handles gridded datasets
    (xarray.Dataset) with optional transformations (spatial subsetting,
    unit conversion, variable selection).

    Built on h5netcdf for pure-Python HDF5 support with no external
    system library dependencies.

    Attributes
    ----------
    output_dir : Path
        Directory where NetCDF files will be written.
    compression : str
        Compression algorithm ('zlib', 'lzf', or None). Default: 'zlib'.
    compression_level : int
        Compression level for zlib (0-9). Default: 4.

    Examples
    --------
    >>> from cosecha import NWPReaper
    >>> from cosecha.sowing.netcdf import NetCDFSower
    >>> reaper = NWPReaper(
    ...     init_time="2026-01-01T00:00:00",
    ...     forecast_hours=[0, 6, 12]
    ... )
    >>> sower = NetCDFSower(output_dir="./data/hrrr")
    >>> harvested = reaper.reap()
    >>> path = sower.sow(harvested)
    >>> print(f"Data written to: {path}")
    """

    def __init__(
        self, output_dir: str | Path, compression: str = "zlib", compression_level: int = 4
    ) -> None:
        """Initialize NetCDFSower.

        Parameters
        ----------
        output_dir : str | Path
            Directory where NetCDF files will be written. Will be created if needed.
        compression : str, optional
            Compression algorithm ('zlib', 'lzf', or None). Default: 'zlib'.
        compression_level : int, optional
            Compression level for zlib (0-9). Default: 4.

        Raises
        ------
        ValueError
            If output_dir is a file (not a directory) or compression is invalid.
        """
        self.output_dir = Path(output_dir)
        self.compression = compression
        self.compression_level = compression_level

        # Validate compression
        if compression not in ("zlib", "lzf", None):
            raise ValueError(f"compression must be 'zlib', 'lzf', or None, got {compression}")

        # Check if path exists as a file
        if self.output_dir.exists() and self.output_dir.is_file():
            raise ValueError(f"output_dir must be a directory, got file: {self.output_dir}")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.debug(
            f"NetCDFSower initialized with output_dir: {self.output_dir}, "
            f"compression: {compression}, compression_level: {compression_level}"
        )

    def _validate_input(self, data: HarvestedData) -> None:
        """Validate input data is suitable for NetCDF writing.

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
                f"NetCDFSower only supports gridded (Dataset) data. "
                f"Got time-series (DataFrame) data with source '{data.source_name}'"
            )
        logger.debug(f"Input validation passed for source: {data.source_name}")

    def sow(
        self,
        data: HarvestedData,
        transformations: Optional[dict[str, Any]] = None,
    ) -> str:
        """Write harvested gridded data to NetCDF file.

        Parameters
        ----------
        data : HarvestedData
            The harvested gridded data to write.
        transformations : dict[str, Any] | None, optional
            Optional transformations to apply before writing. By default None.

        Returns
        -------
        str
            Path to the written NetCDF file.

        Raises
        ------
        SowerError
            If writing fails or data is invalid.
        WriteError
            If the NetCDF write operation fails.
        """
        try:
            self._validate_input(data)

            # Transform if needed
            ds_to_transform = data.data
            import typing
            ds_to_transform = typing.cast(xr.Dataset, ds_to_transform)
            ds = apply_gridded_transformations(ds_to_transform, transformations=transformations)

            # Construct filename from source name and timestamp
            filename = (
                f"{data.source_name.lower()}_" f"{data.timestamp.strftime('%Y%m%d_%H%M%S')}.nc"
            )
            output_path = self.output_dir / filename

            # Write to NetCDF using xarray with h5netcdf engine
            ds.to_netcdf(output_path, mode="w", engine="h5netcdf")

            logger.info(f"Written NetCDF file to {output_path}")
            return str(output_path)

        except SowerError:
            raise
        except Exception as e:
            logger.error(f"NetCDF write failed: {e}")
            raise WriteError(f"NetCDF write failed: {e}") from e
