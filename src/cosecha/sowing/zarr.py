"""ZarrSower for writing gridded data to Zarr format.

This module implements Zarr-based data writing for gridded data
(e.g., HRRR NWP forecasts). Zarr provides cloud-native chunked storage
with efficient compression and support for on-demand data access.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import xarray as xr

from cosecha.data_models import HarvestedData
from cosecha.sowing.base import DataSower
from cosecha.sowing.exceptions import SowerError, WriteError

__all__ = ["ZarrSower"]

logger = logging.getLogger(__name__)


class ZarrSower:
    """Write gridded data to Zarr format.

    Zarr is a cloud-native storage format for chunked, compressed arrays.
    This sower handles gridded datasets (xarray.Dataset) with optional
    transformations (spatial subsetting, unit conversion, resampling).

    Attributes
    ----------
    output_dir : Path
        Directory where Zarr stores will be written.
    consolidate : bool
        Whether to consolidate metadata after writing.

    Examples
    --------
    >>> from cosecha import HRRRReaper, ZarrSower
    >>> reaper = HRRRReaper(
    ...     model="hrrr",
    ...     init_time="2026-01-01 00:00",
    ...     forecast_hours=[0, 6, 12]
    ... )
    >>> sower = ZarrSower(output_dir="./data/hrrr")
    >>> harvested = reaper.reap()
    >>> path = sower.sow(harvested)
    >>> print(f"Data written to: {path}")
    """

    def __init__(self, output_dir: str | Path, consolidate: bool = True) -> None:
        """Initialize ZarrSower.

        Parameters
        ----------
        output_dir : str | Path
            Directory where Zarr stores will be written. Will be created if needed.
        consolidate : bool, optional
            Whether to consolidate metadata after writing (default: True).
            Consolidation improves performance for reads but adds write time.

        Raises
        ------
        ValueError
            If output_dir is a file (not a directory).
        """
        self.output_dir = Path(output_dir)
        self.consolidate = consolidate

        # Check if path exists as a file
        if self.output_dir.exists() and self.output_dir.is_file():
            raise ValueError(f"output_dir must be a directory, got file: {self.output_dir}")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.debug(
            f"ZarrSower initialized with output_dir: {self.output_dir}, "
            f"consolidate: {consolidate}"
        )

    def _validate_input(self, data: HarvestedData) -> None:
        """Validate input data is suitable for Zarr writing.

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
                f"ZarrSower only supports gridded (Dataset) data. "
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
                raise SowerError(f"Spatial subset failed: coordinate not found: {e}") from e

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
            keep = transformations["keep_variables"]
            missing = set(keep) - set(result.data_vars)
            if missing:
                raise SowerError(
                    f"Variables {missing} not found for filtering. "
                    f"Available: {list(result.data_vars)}"
                )
            result = result[keep]
            logger.debug(f"Filtered to variables: {keep}")

        return result

    def sow(self, data: HarvestedData, transformations: Optional[dict[str, Any]] = None) -> str:
        """Write HarvestedData to Zarr store.

        Parameters
        ----------
        data : HarvestedData
            The harvested data to write.
        transformations : dict[str, Any], optional
            Optional transformations to apply before writing.

        Returns
        -------
        str
            The absolute path to the written Zarr store.

        Raises
        ------
        SowerError
            If data validation fails or transformations are invalid.
        WriteError
            If writing to Zarr fails (I/O error, disk space, etc.).

        Examples
        --------
        >>> sower = ZarrSower("./output")
        >>> harvested = reaper.reap()
        >>> # Write with spatial subsetting
        >>> path = sower.sow(
        ...     harvested,
        ...     transformations={
        ...         'spatial_subset': {
        ...             'lon_bounds': [-100, -95],
        ...             'lat_bounds': [30, 35]
        ...         },
        ...         'unit_conversions': {'precip_mm': 0.001}
        ...     }
        ... )
        """
        logger.info(
            f"Sowing {data.source_name} data to Zarr: "
            f"Dataset with {len(data.data.data_vars)} variables"
        )

        try:
            # Validate input
            self._validate_input(data)

            # Get Dataset
            ds = data.data

            # Apply transformations
            ds = self._apply_transformations(ds, transformations)

            # Generate output store name
            source_clean = data.source_name.lower().replace(" ", "_")
            timestamp_str = data.timestamp.strftime("%Y%m%d_%H%M%S")
            store_name = f"{source_clean}_{timestamp_str}.zarr"
            output_path = self.output_dir / store_name

            # Write to Zarr - xarray 2026.2.0+ has full zarr 3.x support
            ds.to_zarr(str(output_path), mode="w")

            logger.info(f"Successfully wrote Zarr store: {output_path}")
            return str(output_path)

        except SowerError:
            raise
        except Exception as e:
            logger.error(f"Failed to write Zarr: {e}")
            raise WriteError(f"Zarr write failed: {e}") from e


# Type hint: ZarrSower implements DataSower protocol
_: type[DataSower] = ZarrSower  # noqa: F841
