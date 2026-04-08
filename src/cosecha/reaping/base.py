"""Base classes for data reapers.

Defines abstract base classes for harvesting and saving data from different source types:
- TimeSeriesReaper: For time-series data like USGS NWIS observations
- GriddedReaper: For gridded data like NWP model output (HRRR, GFS, etc.)
"""

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import overload
from urllib.parse import urlparse

import fsspec
import icechunk
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr
from pyiceberg.catalog import load_catalog

from cosecha._logging import logger
from cosecha._utils import ensure_local_parent, is_remote, wrap_errors
from cosecha.exceptions import ReaperError

__all__ = ["GriddedReaper", "TimeSeriesReaper"]


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
            Harvested data from the source.
        """
        self.data = self._reap()
        return self.data

    @overload
    def _ensure_data(self, expected_type: type[pd.DataFrame], type_label: str) -> pd.DataFrame: ...
    @overload
    def _ensure_data(self, expected_type: type[xr.Dataset], type_label: str) -> xr.Dataset: ...
    def _ensure_data(
        self,
        expected_type: type[pd.DataFrame | xr.Dataset],
        type_label: str,
    ) -> pd.DataFrame | xr.Dataset:
        """Verify that data has been reaped and matches the expected type.

        Called by ``sow_to_*`` methods before writing.

        Parameters
        ----------
        expected_type : type
            Expected type of ``self.data`` (e.g., ``pd.DataFrame``, ``xr.Dataset``).
        type_label : str
            Human-readable label used in error messages.

        Returns
        -------
        pd.DataFrame | xr.Dataset
            The validated data, narrowed to the expected type.

        Raises
        ------
        ReaperError
            If ``reap()`` has not been called or data is not the expected type.
        """
        if self.data is None:
            raise ReaperError("No data to sow. Call reap() first.")
        if not isinstance(self.data, expected_type):
            raise ReaperError(f"Data is not {type_label}.")
        return self.data


class TimeSeriesReaper(ReaperBase):
    """Abstract base class for harvesting time-series data."""

    def sow_to_parquet(self, file_path: str | Path) -> str:
        """Write HarvestedData to Parquet format.

        Parameters
        ----------
        file_path : str | Path
            Local path or remote URI (e.g. ``s3://bucket/key.parquet``)
            where the Parquet file will be written.
            Parent directories are created automatically for local paths.

        Raises
        ------
        ReaperError
            If `reap()` has not been called, or if data is not time-series.

        Returns
        -------
        str
            The path or URI of the written Parquet file.
        """
        data = self._ensure_data(pd.DataFrame, "time-series (DataFrame)")

        path_str = str(file_path)
        with wrap_errors(ReaperError, "Parquet write failed"):
            ensure_local_parent(path_str)
            table = pa.Table.from_pandas(data)
            with fsspec.open(path_str, "wb") as f:
                pq.write_table(table, f, compression="snappy")
            logger.info(f"Successfully wrote Parquet file: {path_str}")
            return path_str

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
        table_name : str
            Name of the Iceberg table to create or append to.
        namespace : str, optional
            Namespace (database) for Iceberg tables (default: 'default').
        catalog_name : str, optional
            Name of the PyIceberg catalog (default: 'default').

        Returns
        -------
        str
            The fully qualified table name (namespace.table_name).
        """
        data = self._ensure_data(pd.DataFrame, "time-series (DataFrame)")

        wh_path = Path(warehouse_path)
        if wh_path.exists() and wh_path.is_file():
            raise ReaperError(f"warehouse_path must be a directory, got file: {wh_path}")
        wh_path.mkdir(parents=True, exist_ok=True)

        metadata_dir = wh_path / ".iceberg"
        metadata_dir.mkdir(parents=True, exist_ok=True)

        with (
            load_catalog(
                catalog_name,
                type="sql",
                uri=f"sqlite:///{metadata_dir / 'iceberg.db'}",
                warehouse=str(wh_path),
                **{"py-io-impl": "pyiceberg.io.fsspec.FsspecFileIO"},
            ) as catalog,
            wrap_errors(ReaperError, "Iceberg write failed"),
        ):
            try:
                table = catalog.load_table(f"{namespace}.{table_name}")
            except Exception:
                with contextlib.suppress(Exception):
                    catalog.create_namespace(namespace)
                pa_table = pa.Table.from_pandas(data)
                table = catalog.create_table(
                    identifier=f"{namespace}.{table_name}", schema=pa_table.schema
                )

            pa_table = pa.Table.from_pandas(data)
            table.append(pa_table)

            full_name = f"{namespace}.{table_name}"
            logger.info(f"Successfully appended data to Iceberg table: {full_name}")
            return full_name


class GriddedReaper(ReaperBase):
    """Abstract base class for harvesting gridded data."""

    def sow_to_zarr(self, file_path: str | Path, consolidate: bool = True) -> str:
        """Write Dataset to Zarr store.

        Parameters
        ----------
        file_path : str | Path
            Local path or remote URI (e.g. ``s3://bucket/store.zarr``)
            where the Zarr store will be written.
            Parent directories are created automatically for local paths.
        consolidate : bool, optional
            Whether to consolidate metadata after writing (default: True).

        Returns
        -------
        str
            The path or URI of the written Zarr store.
        """
        data = self._ensure_data(xr.Dataset, "an xarray Dataset")

        path_str = str(file_path)
        ensure_local_parent(path_str)

        with wrap_errors(ReaperError, "Zarr write failed"):
            data.to_zarr(path_str, mode="w", consolidated=consolidate)
            logger.info(f"Successfully wrote Zarr store: {path_str}")
            return path_str

    def sow_to_netcdf(
        self,
        file_path: str | Path,
    ) -> str:
        """Write Dataset to NetCDF format.

        Parameters
        ----------
        file_path : str | Path
            Path where the NetCDF file will be written. Parent directories will be created if needed.

        Returns
        -------
        str
            Path to the written NetCDF file.
        """
        data = self._ensure_data(xr.Dataset, "an xarray Dataset")

        out_path = Path(file_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with wrap_errors(ReaperError, "NetCDF write failed"):
            data.to_netcdf(out_path, mode="w", engine="h5netcdf")
            logger.info(f"Written NetCDF file to {out_path}")
            return str(out_path)

    def sow_to_icechunk(self, storage_path: str | Path, group_path: str) -> str:
        """Write Dataset to IceChunk format.

        Parameters
        ----------
        storage_path : str | Path
            Local path or S3 URI (e.g. ``s3://bucket/prefix``) for the
            IceChunk storage.  Local directories are created if needed;
            S3 credentials are read from the environment.
        group_path : str
            Path to the IceChunk group within the repository.

        Returns
        -------
        str
            The path or URI of the written IceChunk grouping.
        """
        data = self._ensure_data(xr.Dataset, "an xarray Dataset")

        path_str = str(storage_path)
        if is_remote(path_str):
            parsed = urlparse(path_str)
            storage = icechunk.s3_storage(
                bucket=parsed.netloc,
                prefix=parsed.path.lstrip("/") or None,
                from_env=True,
            )
        else:
            st_path = Path(path_str)
            if st_path.exists() and st_path.is_file():
                raise ReaperError(f"storage_path must be a directory, got file: {st_path}")
            st_path.mkdir(parents=True, exist_ok=True)
            storage = icechunk.local_filesystem_storage(str(st_path))

        try:
            repo = icechunk.Repository.open(storage)
            logger.debug("Opened existing IceChunk repository")
        except Exception:
            repo = icechunk.Repository.create(storage)
            logger.debug("Created new IceChunk repository")

        with wrap_errors(ReaperError, "IceChunk write failed"):
            session = repo.writable_session("main")
            data.to_zarr(
                store=session.store, group=group_path, mode="w", zarr_format=3, consolidated=False
            )
            session.commit(f"Appended data to {group_path}")
            logger.info(f"Successfully committed to IceChunk repo at group: {group_path}")
            return f"{path_str.rstrip('/')}/{group_path}"
