"""Base Protocols for data reapers.

Defines abstract interfaces for harvesting data from different source types:
- TimeSeriesReaper: For time-series data like USGS NWIS observations
- GriddedReaper: For gridded data like NWP model output (HRRR, GFS, etc.)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from cosecha.data_models import HarvestedData

__all__ = ["GriddedReaper", "TimeSeriesReaper"]

logger = logging.getLogger(__name__)


class TimeSeriesReaper(Protocol):
    """Protocol for harvesting time-series data.

    Time-series reapers fetch observational or model data at specific
    station/site locations over a time period, returning data as a
    pd.DataFrame in HarvestedData.

    Methods
    -------
    reap() -> HarvestedData
        Fetch and return time-series data.
    _validate_params() -> None
        Validate input parameters.

    Notes
    -----
    Implementations should:
    - Accept site IDs and date ranges in __init__
    - Validate parameters in _validate_params()
    - Return HarvestedData with DataFrame
    - Log all fetch operations
    - Implement exponential backoff for API retries
    - Raise custom exceptions on failure

    Examples
    --------
    >>> from cosecha.reaping import USGSStreamflowReaper
    >>> reaper = USGSStreamflowReaper(
    ...     site_ids=["01018035"],
    ...     start_date="2026-01-01",
    ...     end_date="2026-01-31",
    ... )
    >>> data = reaper.reap()
    >>> data.is_timeseries()
    True
    """

    def reap(self) -> HarvestedData:
        """Fetch time-series data from source.

        Returns
        -------
        HarvestedData
            Harvested data with pd.DataFrame and metadata.
        """
        ...

    def _validate_params(self) -> None:
        """Validate initialization parameters.

        Raises
        ------
        ValueError
            If parameters are invalid.
        """
        ...


class GriddedReaper(Protocol):
    """Protocol for harvesting gridded data.

    Gridded reapers fetch model output or satellite data on a regular grid
    (e.g., latitude/longitude) over a forecast or analysis period, returning
    data as an xr.Dataset in HarvestedData.

    Methods
    -------
    reap() -> HarvestedData
        Fetch and return gridded data.
    _validate_params() -> None
        Validate input parameters.

    Notes
    -----
    Implementations should:
    - Accept initialization time and forecast/analysis period in __init__
    - Support optional spatial bounds filtering
    - Validate parameters in _validate_params()
    - Return HarvestedData with xarray Dataset
    - Log all fetch operations
    - Return raw data; transformations (subsetting, unit conversion) applied in sowers

    Examples
    --------
    >>> from cosecha.reaping.nwp import NWPReaper
    >>> reaper = NWPReaper(
    ...     model="hrrr",
    ...     init_time="2026-03-20 12:00",
    ...     forecast_hours=range(1, 19),
    ... )
    >>> data = reaper.reap()
    >>> data.is_gridded()
    True
    """

    def reap(self) -> HarvestedData:
        """Fetch gridded data from source.

        Returns
        -------
        HarvestedData
            Harvested data with xr.Dataset and metadata.
        """
        ...

    def _validate_params(self) -> None:
        """Validate initialization parameters.

        Raises
        ------
        ValueError
            If parameters are invalid.
        """
        ...
