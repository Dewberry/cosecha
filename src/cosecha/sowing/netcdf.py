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
    >>> from cosecha import HRRRReaper
    >>> from cosecha.sowing.netcdf import NetCDFSower
    >>> reaper = HRRRReaper(
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

    def _apply_transformations(
        self, ds: xr.Dataset, transformations: Optional[dict[str, Any]] = None
    ) -> xr.Dataset:
        """Apply optional transformations to the xarray Dataset.

        Supported transformations:
        - 'spatial_subset': dict with 'lon_bounds' (min, max) and 'lat_bounds' (min, max)
        - 'unit_conversions': dict mapping variable names to conversion factors
        - 'variable_rename': dict mapping old variable names to new names
        - 'keep_variables': list of variables to keep

        Parameters
        ----------
        ds : xr.Dataset
            The Dataset to transform.
        transformations : dict[str, Any], optional
            Dictionary of transformations to apply. If None, no transformations.

        Returns
        -------
        xr.Dataset
            Transformed Dataset.

        Raises
        ------
        SowerError
            If transformations reference non-existent variables or coordinates.
        """
        if not transformations:
            return ds

        result = ds.copy()

        # Spatial subset
        if "spatial_subset" in transformations:
            subset = transformations["spatial_subset"]
            try:
                if "lon_bounds" in subset:
                    lon_min, lon_max = subset["lon_bounds"]
                    result = result.sel(lon=slice(lon_min, lon_max))
                    logger.debug(f"Applied longitude subset: [{lon_min}, {lon_max}]")

                if "lat_bounds" in subset:
                    lat_min, lat_max = subset["lat_bounds"]
                    result = result.sel(lat=slice(lat_min, lat_max))
                    logger.debug(f"Applied latitude subset: [{lat_min}, {lat_max}]")
            except KeyError as e:
                raise SowerError(
                    f"Spatial subset failed: {e}. Available dimensions: {list(ds.dims)}"
                )

        # Unit conversions
        if "unit_conversions" in transformations:
            conversions = transformations["unit_conversions"]
            for var, factor in conversions.items():
                if var not in result.data_vars:
                    raise SowerError(
                        f"Variable '{var}' not found for unit conversion. "
                        f"Available: {list(result.data_vars)}"
                    )
                result[var] = result[var] * factor
                logger.debug(f"Applied unit conversion to '{var}': factor={factor}")

        # Variable rename
        if "variable_rename" in transformations:
            renames = transformations["variable_rename"]
            result = result.rename(renames)
            logger.debug(f"Renamed variables: {renames}")

        # Keep specific variables
        if "keep_variables" in transformations:
            vars_to_keep = transformations["keep_variables"]
            missing = set(vars_to_keep) - set(result.data_vars)
            if missing:
                raise SowerError(
                    f"Variables to keep not found: {missing}. "
                    f"Available: {list(result.data_vars)}"
                )
            result = result[vars_to_keep]
            logger.debug(f"Kept variables: {vars_to_keep}")

        return result

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
            ds = self._apply_transformations(data.data, transformations=transformations)

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
