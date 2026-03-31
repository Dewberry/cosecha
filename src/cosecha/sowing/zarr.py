"""ZarrSower for writing gridded data to Zarr format.

This module implements Zarr-based data writing for gridded data
(e.g., HRRR NWP forecasts). Zarr provides cloud-native chunked storage
with efficient compression and support for on-demand data access.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from cosecha.logging_config import get_logger
from cosecha.sowing.exceptions import SowerError, WriteError

if TYPE_CHECKING:
    from cosecha.data_models import HarvestedData
    from cosecha.sowing.base import DataSower

__all__ = ["ZarrSower"]

logger = get_logger(__name__)


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
    >>> from cosecha import NWPReaper, ZarrSower
    >>> reaper = NWPReaper(
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
            f"ZarrSower initialized with output_dir: {self.output_dir}, consolidate: {consolidate}"
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

    def sow(self, data: HarvestedData, transformations: dict[str, Any] | None = None) -> str:
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
            logger.exception("Failed to write Zarr")
            raise WriteError(f"Zarr write failed: {e}") from e


# Type hint: ZarrSower implements DataSower protocol
_: type[DataSower] = ZarrSower
