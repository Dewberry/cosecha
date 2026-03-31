"""Base Protocol for data sowers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from cosecha.logging_config import get_logger

if TYPE_CHECKING:
    from cosecha.data_models import HarvestedData

__all__ = ["DataSower"]

logger = get_logger(__name__)


class DataSower(Protocol):
    """Protocol for writing harvested data to storage.

    Sowers accept HarvestedData (which can be time-series DataFrames or
    gridded xarray Datasets) and write them to target storage formats
    (Parquet, Zarr, Iceberg, IceChunk, etc.).

    Methods
    -------
    sow(data: HarvestedData, transformations: dict | None = None) -> str
        Write data to storage with optional transformations.
    _validate_input(data: HarvestedData) -> None
        Validate data before writing.
    _apply_transformations(
        data: HarvestedData,
        transformations: dict[str, Any],
    ) -> HarvestedData
        Apply transformations (e.g., unit conversion, subsetting).

    Notes
    -----
    Implementations should:
    - Validate input data in _validate_input()
    - Apply optional transformations before write
    - Log all write operations
    - Return the path to written data
    - Raise custom exceptions on failure
    - Support append mode where applicable (e.g., Iceberg, Zarr)

    Examples
    --------
    >>> from cosecha.reaping import USGSStreamflowReaper
    >>> from cosecha.sowing.parquet import ParquetSower
    >>> reaper = USGSStreamflowReaper(
    ...     site_ids=["01018035"],
    ...     start_date="2026-01-01",
    ...     end_date="2026-01-31",
    ... )
    >>> data = reaper.reap()
    >>> sower = ParquetSower(store_path="./data/streamflow.parquet")
    >>> path = sower.sow(data)
    """

    def sow(
        self,
        data: HarvestedData,
        transformations: dict[str, Any] | None = None,
    ) -> str:
        """Write harvested data to storage.

        Parameters
        ----------
        data : HarvestedData
            The data to write.
        transformations : dict[str, Any] | None, optional
            Optional transformations to apply before writing
            (e.g., unit conversion, spatial subsetting). By default None.

        Returns
        -------
        str
            Path to the written data.

        Raises
        ------
        SowerError
            If writing fails.
        TransformationError
            If transformations fail.
        """
        ...

    def _validate_input(self, data: HarvestedData) -> None:
        """Validate data before writing.

        Parameters
        ----------
        data : HarvestedData
            The data to validate.

        Raises
        ------
        SowerError
            If data is invalid.
        """
        ...

    def _apply_transformations(
        self,
        data: HarvestedData,
        transformations: dict[str, Any],
    ) -> HarvestedData:
        """Apply transformations to harvested data.

        Parameters
        ----------
        data : HarvestedData
            The data to transform.
        transformations : dict[str, Any]
            Transformation specifications. Exact format depends
            on the sower implementation.

        Returns
        -------
        HarvestedData
            Transformed data.

        Raises
        ------
        TransformationError
            If transformations fail.
        """
        ...
