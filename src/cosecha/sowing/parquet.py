"""ParquetSower for writing time-series data to Parquet format.

This module implements Parquet-based data writing for time-series observations
(e.g., USGS NWIS streamflow, stage, precipitation data).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq

from cosecha.sowing.exceptions import SowerError, WriteError

if TYPE_CHECKING:
    from cosecha.data_models import HarvestedData
    from cosecha.sowing.base import DataSower

__all__ = ["ParquetSower"]

logger = logging.getLogger(__name__)


class ParquetSower:
    """Write time-series data to Parquet format.

    Parquet is an efficient columnar storage format suitable for time-series
    observations. This sower handles transformations (unit conversion, column
    renaming, filtering) and writes HarvestedData to Parquet files.

    Attributes
    ----------
    output_dir : Path
        Directory where Parquet files will be written.

    Examples
    --------
    >>> from cosecha import USGSStreamflowReaper, ParquetSower
    >>> reaper = USGSStreamflowReaper(
    ...     site_ids=["01650000"],
    ...     start_date="2026-01-01",
    ...     end_date="2026-01-31"
    ... )
    >>> sower = ParquetSower(output_dir="./data/streamflow")
    >>> harvested = reaper.reap()
    >>> path = sower.sow(harvested)
    >>> print(f"Data written to: {path}")
    """

    def __init__(self, output_dir: str | Path) -> None:
        """Initialize ParquetSower.

        Parameters
        ----------
        output_dir : str | Path
            Directory where Parquet files will be written. Will be created
            if it does not exist.

        Raises
        ------
        ValueError
            If output_dir is a file (not a directory).
        """
        self.output_dir = Path(output_dir)

        # Check if path exists as a file
        if self.output_dir.exists() and self.output_dir.is_file():
            raise ValueError(f"output_dir must be a directory, got file: {self.output_dir}")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.debug(f"ParquetSower initialized with output_dir: {self.output_dir}")

    def _validate_input(self, data: HarvestedData) -> None:
        """Validate input data is suitable for Parquet writing.

        Parameters
        ----------
        data : HarvestedData
            The harvested data to validate.

        Raises
        ------
        SowerError
            If data is not a DataFrame (gridded data not supported for Parquet).
        """
        if not data.is_timeseries():
            raise SowerError(
                f"ParquetSower only supports time-series (DataFrame) data. "
                f"Got gridded (Dataset) data with source '{data.source_name}'"
            )
        logger.debug(f"Input validation passed for source: {data.source_name}")

    def sow(self, data: HarvestedData) -> str:
        """Write HarvestedData to Parquet file.

        Parameters
        ----------
        data : HarvestedData
            The harvested data to write.

        Returns
        -------
        str
            The absolute path to the written Parquet file.

        Raises
        ------
        SowerError
            If data validation fails or transformations are invalid.
        WriteError
            If writing to Parquet fails (I/O error, disk space, etc.).

        Examples
        --------
        >>> sower = ParquetSower("./output")
        >>> harvested = reaper.reap()
        >>> path = sower.sow(harvested)
        >>> print(f"Data written to: {path}")
        """
        logger.info(f"Sowing {data.source_name} data to Parquet: {len(data.data)} rows")

        try:
            # Validate input
            self._validate_input(data)

            # Get DataFrame
            df = data.data

            # Generate output filename from metadata
            source_clean = data.source_name.lower().replace(" ", "_")
            timestamp_str = data.timestamp.strftime("%Y%m%d_%H%M%S")
            filename = f"{source_clean}_{timestamp_str}.parquet"
            output_path = self.output_dir / filename

            # Write to Parquet
            table = pa.Table.from_pandas(df)
            pq.write_table(table, str(output_path), compression="snappy")

            logger.info(f"Successfully wrote Parquet file: {output_path}")
            return str(output_path)

        except SowerError:
            raise
        except Exception as e:
            logger.exception(f"Failed to write Parquet: {e}")
            raise WriteError(f"Parquet write failed: {e}") from e


# Type hint: ParquetSower implements DataSower protocol
_: type[DataSower] = ParquetSower
