"""Base classes for data reapers.

Defines abstract base classes for harvesting and saving data from different source types:
- TimeSeriesReaper: For time-series data like USGS NWIS observations
- GriddedReaper: For gridded data like NWP model output (HRRR, GFS, etc.)
"""

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from pathlib import Path

import icechunk
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr
from pyiceberg.catalog import load_catalog

from cosecha.exceptions import ReaperError
from cosecha.logging_config import get_logger

__all__ = ["GriddedReaper", "TimeSeriesReaper"]

logger = get_logger(__name__)


class ReaperBase(ABC):
    """Base class for all reapers."""

    def __init__(self) -> None:
        self.data: pd.DataFrame | xr.Dataset | None = None

    @abstractmethod
    def _reap(self) -> pd.DataFrame | xr.Dataset:
        """Implement data fetching."""

    @abstractmethod
    def _validate_params(self) -> None:
        """Validate initialization parameters."""

    def reap(self) -> pd.DataFrame | xr.Dataset:
        """Fetch data from source and store it in instance state.

        Returns
        -------
        pd.DataFrame | xr.Dataset
            Harvested data with metadata in attrs.
        """
        self.data = self._reap()
        return self.data


class TimeSeriesReaper(ReaperBase):
    """Abstract base class for harvesting time-series data."""

    def sow_to_parquet(self, file_path: str | Path) -> str:
        """Write HarvestedData to Parquet format.

        Parameters
        ----------
        file_path : str | Path
            The file path where the Parquet file will be written. Parent directories will be created if needed.

        Raises
        ------
        ReaperError
            If `reap()` has not been called, or if data is not time-series.
            If writing to Parquet fails.

        Returns
        -------
        str
            The absolute path to the written Parquet file.
        """
        if self.data is None:
            raise ReaperError("No data to sow. Call reap() first.")

        if not isinstance(self.data, pd.DataFrame):
            raise ReaperError("Data is not time-series (DataFrame).")

        out_path = Path(file_path)

        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            df = self.data

            output_path = out_path

            table = pa.Table.from_pandas(df)

            pq.write_table(table, str(output_path), compression="snappy")

            logger.info(f"Successfully wrote Parquet file: {output_path}")
            return str(output_path)
        except Exception as e:
            logger.exception("Failed to write Parquet")
            raise ReaperError(f"Parquet write failed: {e}") from e

    def sow_to_iceberg(
        self,
        warehouse_path: str | Path,
        table_name: str,
        namespace: str = "default",
        catalog_name: str = "default",
    ) -> str:
        """Write tabular data to Apache Iceberg format.

        Parameters
        ----------
        warehouse_path : str | Path
            Path to the Iceberg warehouse directory. Will be created if needed.
        namespace : str, optional
            Namespace (database) for Iceberg tables (default: 'default').
        catalog_name : str, optional
            Name of the PyIceberg catalog (default: 'default').

        Returns
        -------
        str
            The fully qualified table name (namespace.table_name).
        """
        if self.data is None:
            raise ReaperError("No data to sow. Call reap() first.")
        if not isinstance(self.data, pd.DataFrame):
            raise ReaperError("Data is not time-series (DataFrame).")

        wh_path = Path(warehouse_path)
        if wh_path.exists() and wh_path.is_file():
            raise ValueError(f"warehouse_path must be a directory, got file: {wh_path}")
        wh_path.mkdir(parents=True, exist_ok=True)

        metadata_dir = wh_path / ".iceberg"
        metadata_dir.mkdir(parents=True, exist_ok=True)

        catalog = load_catalog(
            catalog_name,
            type="sql",
            uri=f"sqlite:///{metadata_dir / 'iceberg.db'}",
            warehouse=str(wh_path),
        )

        try:
            df = self.data

            try:
                table = catalog.load_table(f"{namespace}.{table_name}")
            except Exception:
                with contextlib.suppress(Exception):
                    catalog.create_namespace(namespace)

                pa_table = pa.Table.from_pandas(df)
                table = catalog.create_table(
                    name=f"{namespace}.{table_name}", schema=pa_table.schema
                )

            pa_table = pa.Table.from_pandas(df)
            table.append(pa_table)

            full_name = f"{namespace}.{table_name}"
            logger.info(f"Successfully appended data to Iceberg table: {full_name}")
        except Exception as e:
            logger.exception("Failed to write to Iceberg")
            raise ReaperError(f"Iceberg write failed: {e}") from e
        else:
            return full_name


class GriddedReaper(ReaperBase):
    """Abstract base class for harvesting gridded data."""

    def sow_to_zarr(self, file_path: str | Path, consolidate: bool = True) -> str:
        """Write Dataset to Zarr store.

        Parameters
        ----------
        file_path : str | Path
            Path where the Zarr store will be written. Parent directories will be created if needed.
        consolidate : bool, optional
            Whether to consolidate metadata after writing (default: True).

        Returns
        -------
        str
            The absolute path to the written Zarr store.
        """
        if self.data is None:
            raise ReaperError("No data to sow. Call reap() first.")
        if not isinstance(self.data, xr.Dataset):
            raise ReaperError("Data is not an xarray Dataset.")

        out_path = Path(file_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            ds = self.data
            output_path = out_path

            ds.to_zarr(str(output_path), mode="w", consolidated=consolidate)

            logger.info(f"Successfully wrote Zarr store: {output_path}")
            return str(output_path)
        except Exception as e:
            logger.exception("Failed to write Zarr")
            raise ReaperError(f"Zarr write failed: {e}") from e

    def sow_to_netcdf(
        self,
        file_path: str | Path,
    ) -> str:
        """Write Dataset to NetCDF format.

        Parameters
        ----------
        file_path : str | Path
            Path where the NetCDF file will be written. Parent directories will be created if needed.
        compression : str, optional
            Compression algorithm ('zlib', 'lzf', or None). Default: 'zlib'.

        Returns
        -------
        str
            Path to the written NetCDF file.
        """
        if self.data is None:
            raise ReaperError("No data to sow. Call reap() first.")
        if not isinstance(self.data, xr.Dataset):
            raise ReaperError("Data is not an xarray Dataset.")

        out_path = Path(file_path)

        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            ds = self.data
            output_path = out_path

            ds.to_netcdf(output_path, mode="w", engine="h5netcdf")

            logger.info(f"Written NetCDF file to {output_path}")
            return str(output_path)
        except Exception as e:
            logger.exception("NetCDF write failed")
            raise ReaperError(f"NetCDF write failed: {e}") from e

    def sow_to_icechunk(self, storage_path: str | Path, group_path: str) -> str:
        """Write Dataset to IceChunk format.

        Parameters
        ----------
        storage_path : str | Path
            Path to the IceChunk storage directory. Will be created if needed.

        Returns
        -------
        str
            The absolute path to the written IceChunk grouping.
        """
        if self.data is None:
            raise ReaperError("No data to sow. Call reap() first.")
        if not isinstance(self.data, xr.Dataset):
            raise ReaperError("Data is not an xarray Dataset.")

        st_path = Path(storage_path)
        if st_path.exists() and st_path.is_file():
            raise ValueError(f"storage_path must be a directory, got file: {st_path}")
        st_path.mkdir(parents=True, exist_ok=True)

        storage = icechunk.local_filesystem_storage(str(st_path))
        try:
            repo = icechunk.Repository.open(storage)
            logger.debug("Opened existing IceChunk repository")
        except Exception:
            repo = icechunk.Repository.create(storage)
            logger.debug("Created new IceChunk repository")

        try:
            ds = self.data
            commit_msg = f"Appended data to {group_path}"

            session = repo.writable_session("main")
            ds.to_zarr(
                store=session.store, group=group_path, mode="w", zarr_format=3, consolidated=False
            )

            session.commit(commit_msg)

            logger.info(f"Successfully committed to IceChunk repo at group: {group_path}")
            return str(st_path / group_path)
        except Exception as e:
            logger.exception("Failed to write to IceChunk")
            raise ReaperError(f"IceChunk write failed: {e}") from e
