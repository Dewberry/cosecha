"""IcebergSower for writing time-series data to Apache Iceberg tables.

This module implements Iceberg-based data writing for time-series observations.
Iceberg tables provide ACID transactions, time-travel, schema evolution, and
hidden partitioning—ideal for maintaining versioned historical data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import pyarrow as pa
from pyiceberg.catalog import Catalog

from cosecha.data_models import HarvestedData
from cosecha.sowing.base import DataSower
from cosecha.sowing.exceptions import SowerError, WriteError

__all__ = ["IcebergSower"]

logger = logging.getLogger(__name__)


class IcebergSower:
    """Write time-series data to Apache Iceberg tables.

    Iceberg tables support ACID transactions, time-travel, and hidden partitioning.
    This sower creates or appends to Iceberg tables, enabling version control
    and historical data retention for time-series observations.

    Uses local file-based catalogs with SqlCatalog and SQLite metadata store,
    requiring no external Iceberg services.

    Attributes
    ----------
    warehouse_path : Path
        Path to the Iceberg warehouse directory.
    namespace : str
        Namespace (database) for Iceberg tables.
    catalog_name : str
        Name of the PyIceberg catalog (default: 'default').

    Examples
    --------
    >>> from cosecha import USGSStreamflowReaper, IcebergSower
    >>> reaper = USGSStreamflowReaper(
    ...     site_ids=["01650000"],
    ...     start_date="2026-01-01",
    ...     end_date="2026-01-31"
    ... )
    >>> sower = IcebergSower(
    ...     warehouse_path="./data/iceberg",
    ...     namespace="nwis"
    ... )
    >>> harvested = reaper.reap()
    >>> table_name = sower.sow(harvested)
    >>> print(f"Data written to: {table_name}")
    """

    def __init__(
        self, warehouse_path: str | Path, namespace: str = "default", catalog_name: str = "default"
    ) -> None:
        """Initialize IcebergSower.

        Parameters
        ----------
        warehouse_path : str | Path
            Path to the Iceberg warehouse directory. Will be created if needed.
        namespace : str, optional
            Namespace (database) for Iceberg tables (default: 'default').
        catalog_name : str, optional
            Name of the PyIceberg catalog (default: 'default').

        Raises
        ------
        ValueError
            If warehouse_path is a file (not a directory).
        """
        self.warehouse_path = Path(warehouse_path)
        self.namespace = namespace
        self.catalog_name = catalog_name

        # Check if path exists as a file
        if self.warehouse_path.exists() and self.warehouse_path.is_file():
            raise ValueError(f"warehouse_path must be a directory, got file: {self.warehouse_path}")

        self.warehouse_path.mkdir(parents=True, exist_ok=True)

        try:
            # Create a file-based catalog
            from pyiceberg.catalog.sql import SqlCatalog

            # Create metadata directory for catalog
            metadata_dir = self.warehouse_path / ".iceberg"
            metadata_dir.mkdir(parents=True, exist_ok=True)

            # Use SqlCatalog with filesystem backend (no external dependencies)
            # This requires sqlalchemy, so fall back to load_catalog approach if not available
            try:
                self.catalog = SqlCatalog(
                    name=catalog_name,
                    **{
                        "uri": f"sqlite:///{metadata_dir / 'iceberg.db'}",
                        "warehouse": str(self.warehouse_path),
                    },
                )
            except ModuleNotFoundError:
                # SqlAlchemy not available, use load_catalog
                from pyiceberg.catalog import load_catalog

                self.catalog = load_catalog(name=catalog_name, warehouse=str(self.warehouse_path))

            logger.debug(
                f"IcebergSower initialized with catalog: {catalog_name}, "
                f"namespace: {namespace}, warehouse: {self.warehouse_path}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Iceberg catalog: {e}")
            raise ValueError(f"Failed to initialize Iceberg catalog: {e}") from e

    def _validate_input(self, data: HarvestedData) -> None:
        """Validate input data is suitable for Iceberg writing.

        Parameters
        ----------
        data : HarvestedData
            The harvested data to validate.

        Raises
        ------
        SowerError
            If data is not a DataFrame (gridded data not supported).
        """
        if not data.is_timeseries():
            raise SowerError(
                f"IcebergSower only supports time-series (DataFrame) data. "
                f"Got gridded (Dataset) data with source '{data.source_name}'"
            )
        logger.debug(f"Input validation passed for source: {data.source_name}")

    def _get_or_create_table(self, table_name: str, df: pd.DataFrame):
        """Get existing table or create new Iceberg table.

        Parameters
        ----------
        table_name : str
            Simple table name (without namespace).
        df : pd.DataFrame
            DataFrame used to infer schema if creating a new table.

        Returns
        -------
        IcebergTable
            The Iceberg table object.

        Raises
        ------
        WriteError
            If table creation or retrieval fails.
        """
        try:
            # Try to load existing table
            table = self.catalog.load_table(f"{self.namespace}.{table_name}")
            logger.debug(f"Loaded existing Iceberg table: {self.namespace}.{table_name}")
            return table
        except Exception:
            # Table doesn't exist, create it
            try:
                # Create namespace if needed
                try:
                    self.catalog.create_namespace(self.namespace)
                    logger.debug(f"Created namespace: {self.namespace}")
                except Exception:
                    # Namespace may already exist
                    pass

                # Convert DataFrame to PyArrow table to infer schema
                pa_table = pa.Table.from_pandas(df)

                # Create Iceberg table
                table = self.catalog.create_table(
                    name=f"{self.namespace}.{table_name}", schema=pa_table.schema
                )
                logger.info(f"Created new Iceberg table: {self.namespace}.{table_name}")
                return table
            except Exception as e:
                raise WriteError(f"Failed to create Iceberg table: {e}") from e

    def sow(self, data: HarvestedData) -> str:
        """Write HarvestedData to Iceberg table.

        Parameters
        ----------
        data : HarvestedData
            The harvested data to write.

        Returns
        -------
        str
            The fully qualified table name (namespace.table_name).

        Raises
        ------
        SowerError
            If data validation fails or transformations are invalid.
        WriteError
            If writing to Iceberg fails.

        Examples
        --------
        >>> sower = IcebergSower(warehouse_path="./data/iceberg")
        >>> harvested = reaper.reap()
        >>> table_name = sower.sow(
        ...     harvested
        ... )
        """
        logger.info(f"Sowing {data.source_name} data to Iceberg: " f"{len(data.data)} rows")

        try:
            # Validate input
            self._validate_input(data)

            # Get DataFrame
            df = data.data

            # Generate table name from source
            table_name = data.source_name.lower().replace(" ", "_")

            # Get or create table
            table = self._get_or_create_table(table_name, df)

            # Convert to PyArrow and append
            pa_table = pa.Table.from_pandas(df)
            table.append(pa_table)

            full_name = f"{self.namespace}.{table_name}"
            logger.info(f"Successfully appended data to Iceberg table: {full_name}")
            return full_name

        except SowerError:
            raise
        except Exception as e:
            logger.error(f"Failed to write to Iceberg: {e}")
            raise WriteError(f"Iceberg write failed: {e}") from e


# Type hint: IcebergSower implements DataSower protocol
_: type[DataSower] = IcebergSower  # noqa: F841
