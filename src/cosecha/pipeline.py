"""Harvest pipeline orchestrator for coordinating reapers and sowers.

This module provides the HarvestPipeline class, which orchestrates the
complete data harvesting workflow: fetch data from a reaper, apply optional
transformations, and write to one or more sowers (storage backends).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from cosecha.reaping.exceptions import ReaperError
from cosecha.sowing.exceptions import SowerError

if TYPE_CHECKING:
    from cosecha.data_models import HarvestedData
    from cosecha.reaping.base import GriddedReaper, TimeSeriesReaper
    from cosecha.sowing.base import DataSower

__all__ = ["HarvestPipeline"]

logger = logging.getLogger(__name__)


class HarvestPipeline:
    """Orchestrate data harvesting from reapers to sowers.

    HarvestPipeline coordinates the complete data harvesting workflow:
    1. Fetch data from a reaper (TimeSeriesReaper or GriddedReaper)
    2. Apply optional transformations
    3. Write harvested data to one or more sowers (storage backends)

    A single pipeline can handle the complete workflow from multiple
    sources (reapers) to multiple storage targets (sowers). Typical usage
    patterns include:
    - Fetch USGS NWIS streamflow → Write to Parquet (time-series)
    - Fetch HRRR forecast grid → Write to Zarr (gridded)
    - Fetch data → Write to both Parquet and Iceberg (backup)

    Attributes
    ----------
    reapers : list[TimeSeriesReaper | GriddedReaper]
        List of reapers to fetch data from.
    sowers : list[DataSower]
        List of sowers to write data to.

    Examples
    --------
    Time-series data pipeline (NWIS → Parquet):

    >>> from cosecha import USGSStreamflowReaper, ParquetSower
    >>> from cosecha.pipeline import HarvestPipeline
    >>> reaper = USGSStreamflowReaper(
    ...     site_ids=["01018035"],
    ...     start_date="2026-01-01",
    ...     end_date="2026-01-31",
    ... )
    >>> sower = ParquetSower(output_dir="./data/streamflow")
    >>> pipeline = HarvestPipeline()
    >>> pipeline.add_reaper(reaper)
    >>> pipeline.add_sower(sower)
    >>> paths = pipeline.execute()
    >>> print(f"Data written to: {paths}")
    ['./data/streamflow/20260101_20260131.parquet']

    Gridded data pipeline (HRRR → Zarr):

    >>> from cosecha import NWPReaper
    >>> from cosecha.sowing.zarr import ZarrSower
    >>> reaper = NWPReaper(init_time="2026-01-01T00:00:00", forecast_hours=[0, 6, 12])
    >>> sower = ZarrSower(output_dir="./data/hrrr")
    >>> pipeline = HarvestPipeline()
    >>> pipeline.add_reaper(reaper)
    >>> pipeline.add_sower(sower)
    >>> paths = pipeline.execute()
    >>> print(f"Data written to: {paths}")
    ['./data/hrrr/20260101T000000.zarr']

    Multi-sink pipeline (same data to Parquet and Iceberg):

    >>> from cosecha import USGSStreamflowReaper, ParquetSower
    >>> from cosecha.sowing.iceberg import IcebergSower
    >>> reaper = USGSStreamflowReaper(
    ...     site_ids=["01018035"],
    ...     start_date="2026-01-01",
    ...     end_date="2026-01-31",
    ... )
    >>> parquet_sower = ParquetSower(output_dir="./data/streamflow")
    >>> iceberg_sower = IcebergSower(output_dir="./data/warehouse")
    >>> pipeline = HarvestPipeline()
    >>> pipeline.add_reaper(reaper)
    >>> pipeline.add_sower(parquet_sower)
    >>> pipeline.add_sower(iceberg_sower)
    >>> paths = pipeline.execute()
    >>> print(f"Data written to: {paths}")
    ['./data/streamflow/...', './data/warehouse/...']

    With transformations (e.g., unit conversion):

    >>> transformations = {
    ...     "unit_conversions": {
    ...         "mean_streamflow": 0.0283168,  # cfs to m³/s
    ...     }
    ... }
    >>> paths = pipeline.execute(transformations=transformations)
    """

    def __init__(self) -> None:
        """Initialize an empty HarvestPipeline.

        The pipeline starts with no reapers or sowers. Use add_reaper()
        and add_sower() to configure the pipeline before calling execute().
        """
        self.reapers: list[TimeSeriesReaper | GriddedReaper] = []
        self.sowers: list[DataSower] = []
        logger.debug("HarvestPipeline initialized")

    def add_reaper(self, reaper: TimeSeriesReaper | GriddedReaper) -> None:
        """Add a reaper to the pipeline.

        Parameters
        ----------
        reaper : TimeSeriesReaper | GriddedReaper
            A reaper implementing either TimeSeriesReaper or GriddedReaper
            protocol to fetch data.

        Raises
        ------
        TypeError
            If reaper does not implement TimeSeriesReaper or GriddedReaper.

        Notes
        -----
        Multiple reapers can be added to the pipeline. When execute() is
        called, each reaper will be called to fetch data, and the data
        will be passed to all sowers.

        However, for typical use cases with uniform transformations and
        homogeneous output formats, a single-reaper-to-multiple-sowers
        pattern is recommended.
        """
        # Validate that reaper implements the required protocol
        if not (hasattr(reaper, "reap") and callable(reaper.reap)):
            raise TypeError(
                f"reaper must implement TimeSeriesReaper or GriddedReaper protocol "
                f"(must have callable `reap()` method), got {type(reaper).__name__}"
            )

        self.reapers.append(reaper)
        logger.debug(f"Added reaper: {type(reaper).__name__}")

    def add_sower(self, sower: DataSower) -> None:
        """Add a sower to the pipeline.

        Parameters
        ----------
        sower : DataSower
            A sower implementing DataSower protocol to write harvested data.

        Raises
        ------
        TypeError
            If sower does not implement DataSower protocol.

        Notes
        -----
        Multiple sowers can be added to the pipeline. When execute() is
        called, harvested data will be written to all sowers. This enables
        patterns like writing to both Parquet (for analysis) and Iceberg
        (for data warehouse) simultaneously.
        """
        # Validate that sower implements the required protocol
        if not (hasattr(sower, "sow") and callable(sower.sow)):
            raise TypeError(
                f"sower must implement DataSower protocol "
                f"(must have callable `sow()` method), got {type(sower).__name__}"
            )

        self.sowers.append(sower)
        logger.debug(f"Added sower: {type(sower).__name__}")

    def execute(self) -> list[str]:
        """Execute the harvest pipeline: reap → sow.

        Orchestrates the complete data harvesting workflow:
        1. Call reap() on each registered reaper
        2. Call sow() on each registered sower with harvested data
        3. Return list of paths/table names from all sowers

        Returns
        -------
        list[str]
            List of paths or table names returned by sowers. Each sower
            returns a string indicating where data was written (e.g.,
            file path, table name, URI).

        Raises
        ------
        ValueError
            If pipeline has no reapers or sowers configured.
        ReaperError
            If reaping fails (network error, invalid parameters, etc.).
        SowerError
            If sowing fails (write error, invalid data, etc.).

        Notes
        -----
        The pipeline is designed for a single-source-to-multiple-sinks
        pattern. While multiple reapers can be added, typical usage
        involves a single reaper. If multiple reapers are used, they will
        be executed sequentially (not in parallel), and all harvested data
        will be passed to all sowers.

        Transformations are applied uniformly to all data before sowing.
        If different sowers need different transformations, consider
        calling sower.sow() directly on harvested data instead of using
        the pipeline's execute() method.

        Examples
        --------
        Execute without transformations:

        >>> pipeline = HarvestPipeline()
        >>> pipeline.add_reaper(reaper)
        >>> pipeline.add_sower(sower)
        >>> paths = pipeline.execute()
        """
        if not self.reapers:
            raise ValueError("Pipeline has no reapers configured. Use add_reaper() first.")
        if not self.sowers:
            raise ValueError("Pipeline has no sowers configured. Use add_sower() first.")

        paths: list[str] = []
        logger.info(
            f"Starting harvest pipeline: {len(self.reapers)} reaper(s), {len(self.sowers)} sower(s)"
        )

        # Execute all reapers
        harvested_data_list: list[HarvestedData] = []
        for reaper in self.reapers:
            try:
                logger.info(f"Reaping data with {type(reaper).__name__}...")
                data = reaper.reap()
                harvested_data_list.append(data)
                logger.info(
                    f"Successfully reaped data from {data.source_name}: "
                    f"{len(data.variable_names)} variables, "
                    f"data type={data._data_type()}"
                )
            except ReaperError as e:
                logger.exception(f"Reaper {type(reaper).__name__} failed: {e}")
                raise
            except Exception as e:
                logger.exception(f"Unexpected error in reaper {type(reaper).__name__}: {e}")
                raise ReaperError(
                    f"Reaper {type(reaper).__name__} failed with unexpected error: {e}"
                ) from e

        # Sow to all sowers
        for data in harvested_data_list:
            for sower in self.sowers:
                try:
                    logger.info(
                        f"Sowing data to {type(sower).__name__}: "
                        f"source={data.source_name}, "
                        f"variables={len(data.variable_names)}"
                    )
                    path = sower.sow(data)
                    paths.append(path)
                    logger.info(f"Successfully sowed data to {type(sower).__name__}: {path}")
                except SowerError as e:
                    logger.exception(f"Sower {type(sower).__name__} failed: {e}")
                    raise
                except Exception as e:
                    logger.exception(f"Unexpected error in sower {type(sower).__name__}: {e}")
                    raise SowerError(
                        f"Sower {type(sower).__name__} failed with unexpected error: {e}"
                    ) from e

        logger.info(
            f"Harvest pipeline completed successfully: wrote data to {len(paths)} output(s)"
        )
        return paths
