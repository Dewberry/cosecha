"""NWP (Numerical Weather Prediction) reapers for gridded forecast data.

This module implements reapers for high-resolution numerical weather prediction
models (HRRR, NAM, etc.) using the herbie library for data fetching.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Optional

import xarray as xr

from cosecha.data_models import HarvestedData
from cosecha.reaping.base import GriddedReaper
from cosecha.reaping.exceptions import APIError, DateRangeError, ReaperError

__all__ = ["HRRRReaper"]

logger = logging.getLogger(__name__)


class HRRRReaper:
    """Fetch High-Resolution Rapid Refresh (HRRR) forecast data.

    The HRRR is a real-time, high-resolution (3-km grid spacing) numerical
    weather prediction model that is updated hourly. This reaper fetches
    raw HRRR data without applying transformations; transformations
    (unit conversion, spatial subsetting, etc.) are handled by sowers.

    Attributes
    ----------
    model : str
        NWP model name (default: 'hrrr').
    init_time : str
        Model initialization time in YYYYMMDDHH format or datetime string.
    forecast_hours : list[int] or range
        Forecast hours to request (e.g., [1, 6, 12] or range(1, 19)).

    Examples
    --------
    >>> from cosecha import HRRRReaper
    >>> reaper = HRRRReaper(
    ...     init_time="2026-01-01 00:00",
    ...     forecast_hours=range(1, 7)  # 1-6 hour forecasts
    ... )
    >>> harvested = reaper.reap()
    >>> print(f"Fetched {len(harvested.data.data_vars)} variables")
    >>> print(f"Grid dimensions: {harvested.data.dims}")

    Notes
    -----
    - Requires herbie >= 2.6.0 to be installed
    - HRRR data is publicly available from NOAA AWS buckets
    - Raw data uses degrees east longitude; conversion to degrees west
      and regional subsetting are handled by ZarrSower or custom pipelines
    """

    def __init__(
        self,
        init_time: str,
        forecast_hours: list[int] | range,
        model: str = "hrrr",
    ) -> None:
        """Initialize HRRRReaper.

        Parameters
        ----------
        init_time : str
            Model initialization time in format "YYYY-MM-DD HH:MM" or similar.
            Will be parsed by pandas.to_datetime().
        forecast_hours : list[int] | range
            Forecast hours to request (e.g., [1, 6, 12] or range(1, 19)).
        model : str, optional
            NWP model name (default: 'hrrr'). Other options: 'nam', 'gfs', etc.

        Raises
        ------
        DateRangeError
            If init_time is invalid or forecast_hours are malformed.
        ValueError
            If herbie is not installed.

        Examples
        --------
        >>> reaper = HRRRReaper(
        ...     init_time="2026-01-01 00:00",
        ...     forecast_hours=range(1, 19)
        ... )
        """
        self.model = model
        self.init_time = init_time
        self.forecast_hours = (
            list(forecast_hours) if isinstance(forecast_hours, range) else forecast_hours
        )

        # Validate parameters
        self._validate_params()

        # Check herbie availability
        try:
            from herbie import FastHerbie

            self._herbie_available = True
        except ImportError:
            logger.warning(
                "herbie not installed; HRRRReaper will not be able to fetch data. "
                "Install with: pip install 'cosecha[nwp]'"
            )
            self._herbie_available = False

        logger.debug(
            f"HRRRReaper initialized: model={model}, init_time={init_time}, "
            f"forecast_hours={len(self.forecast_hours)}"
        )

    def _validate_params(self) -> None:
        """Validate initialization parameters.

        Raises
        ------
        DateRangeError
            If init_time cannot be parsed or forecast_hours are invalid.
        """
        import pandas as pd

        # Validate init_time
        try:
            parsed_time = pd.to_datetime(self.init_time)
            logger.debug(f"Parsed init_time: {parsed_time}")
        except Exception as e:
            raise DateRangeError(f"Could not parse init_time '{self.init_time}': {e}") from e

        # Validate forecast_hours
        if not self.forecast_hours or len(self.forecast_hours) == 0:
            raise DateRangeError("forecast_hours cannot be empty")

        if not all(isinstance(h, int) and h > 0 for h in self.forecast_hours):
            raise DateRangeError(
                f"forecast_hours must be positive integers, got {self.forecast_hours}"
            )

        logger.debug(f"Validation passed for forecast_hours: {self.forecast_hours}")

    def _fetch_with_herbie(self) -> xr.Dataset:
        """Fetch raw HRRR data using FastHerbie.

        Returns
        -------
        xr.Dataset
            Raw HRRR data with all requested forecast hours and variables.

        Raises
        ------
        APIError
            If herbie fails to fetch data.
        """
        if not self._herbie_available:
            raise APIError("herbie is not installed. Install with: pip install 'cosecha[nwp]'")

        try:
            from herbie import FastHerbie

            logger.info(
                f"Fetching HRRR data: model={self.model}, init_time={self.init_time}, "
                f"forecast_hours={self.forecast_hours}"
            )

            # Use default filter (all variables) or a specific filter if desired
            # Users can customize via transformations in sowers
            h = FastHerbie([self.init_time], model=self.model, fxx=self.forecast_hours)

            # Fetch all variables; filtering is a sower concern
            ds = h.xarray()

            logger.info(f"Successfully fetched HRRR data: {len(ds.data_vars)} variables")
            return ds

        except Exception as e:
            logger.error(f"Failed to fetch HRRR data: {e}")
            raise APIError(f"HRRR fetch failed: {e}") from e

    def reap(self) -> HarvestedData:
        """Fetch and return HRRR forecast data.

        Returns
        -------
        HarvestedData
            Container with xarray Dataset, source name, timestamp, variables,
            and metadata including init_time, forecast_hours, model.

        Raises
        ------
        APIError
            If data fetching fails.

        Examples
        --------
        >>> reaper = HRRRReaper(
        ...     init_time="2026-01-01 00:00",
        ...     forecast_hours=range(1, 7)
        ... )
        >>> harvested = reaper.reap()
        >>> print(harvested.data)  # xarray Dataset
        >>> print(harvested.metadata)  # dict with init_time, forecast_hours, etc.
        """
        logger.info(f"Reaping HRRR data for {self.model} model")

        try:
            # Fetch raw data
            ds = self._fetch_with_herbie()

            # Ensure all dimension names are lowercase (xarray convention)
            ds = ds.rename({k: k.lower() for k in ds.dims if k != k.lower()})

            # Get variable names
            variable_names = list(ds.data_vars)

            # Create metadata
            metadata = {
                "source_name": "HRRR Forecast",
                "timestamp": datetime.now(UTC),
                "variable_names": variable_names,
                "model": self.model,
                "init_time": self.init_time,
                "forecast_hours": self.forecast_hours,
                "nwp_grid_points": ds.sizes.get("x", 0) * ds.sizes.get("y", 0),
            }

            # Create HarvestedData
            harvested = HarvestedData(
                data=ds,
                source_name="HRRR Forecast",
                timestamp=metadata["timestamp"],
                variable_names=variable_names,
                metadata=metadata,
            )

            logger.info(
                f"Successfully reaped HRRR data: {len(variable_names)} variables, "
                f"{len(self.forecast_hours)} forecast hours"
            )
            return harvested

        except APIError:
            raise
        except Exception as e:
            logger.error(f"Failed to reap HRRR data: {e}")
            raise ReaperError(f"HRRR reaping failed: {e}") from e


# Type hint: HRRRReaper implements GriddedReaper protocol
_: type[GriddedReaper] = HRRRReaper  # noqa: F841
