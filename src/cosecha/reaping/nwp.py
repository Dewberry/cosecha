"""NWP (Numerical Weather Prediction) reapers for gridded forecast data.

This module implements reapers for high-resolution numerical weather prediction
models (HRRR, RRFS, etc.) using the herbie library for data fetching.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pandas as pd

try:
    from herbie import FastHerbie

    _HERBIE_AVAILABLE = True
except ImportError:
    FastHerbie = None
    _HERBIE_AVAILABLE = False

from cosecha.data_models import HarvestedData
from cosecha.reaping.exceptions import APIError, DateRangeError, ReaperError
from cosecha.logging_config import get_logger
from cosecha.reaping.utils import apply_gridded_transformations

if TYPE_CHECKING:
    import xarray as xr

    from cosecha.reaping.base import GriddedReaper

__all__ = ["NWPReaper"]

logger = get_logger(__name__)

NWP_SEARCH_STRINGS = {
    "hrrr": {
        "hourly_precip": r":APCP:.*:(?:0-1|[1-9]\d*-\d+) hour",
        "total_precip": r":APCP:surface:0-[1-9]*",
        "temp_2m": r"TMP:2 m above",
    },
    "rrfs": {
        "hourly_precip": r":APCP:.*:(?:0-1|[1-9]\d*-\d+) hour",
        "total_precip": r":APCP:surface:0-[1-9]*",
        "temp_2m": r":TMP:2 m above ground:",
    },
    "rtma": {
        "temp_2m": r"TMP:2 m above ground",
    },
    # Add other models and variables as needed
}


class NWPReaper:
    """Fetch NOAA Numerical Weather Prediction (NWP) forecast data."""

    def _validate_params(self) -> None:
        """Validate initialization parameters.

        Raises
        ------
        DateRangeError
            If init_time cannot be parsed or forecast_hours are invalid.
        """
        # Validate init_time
        try:
            parsed_time = pd.to_datetime(self.init_time)
            logger.debug(f"Parsed init_time: {parsed_time}")
        except Exception as e:
            raise DateRangeError(f"Could not parse init_time '{self.init_time}': {e}") from e

        if self.forecast_hours is not None and not all(
            isinstance(h, int) and h > 0 for h in self.forecast_hours
        ):
            raise DateRangeError(
                f"forecast_hours must be positive integers, got {self.forecast_hours}"
            )

        logger.debug(f"Validation passed for forecast_hours: {self.forecast_hours}")

    def __init__(
        self,
        init_time: str,
        forecast_hours: list[int] | range | None = None,
        model: str = "hrrr",
        variable: str | None = "hourly_precip",
        search_str: str | None = None,
        product: str | None = None,
        transformations: dict[str, Any] | None = None,
    ) -> None:
        """Initialize NWPReaper.

        Parameters
        ----------
        init_time : str
            Model initialization time in format "YYYY-MM-DD HH:MM" or similar.
            Will be parsed by pandas.to_datetime().
        forecast_hours : list[int] | range | None, optional
            Forecast hours to request (e.g., [1, 6, 12] or range(1, 19)). Can be none if fetching analysis product.
        model : str, optional
            NWP model name (default: 'hrrr'). Other options: 'rrfs', 'rtma', etc.
        variable: str, optional
            A simplified variable name mapping to a predefined GRIB regex search string. Common examples include 'hourly_precip', 'total_precip', 'temp_2m'. Ignored if `search_str` is provided.
        search_str: str, optional
            Exact GRIB regex search string to use. Overrides the `variable` lookup if provided.
        product: str, optional
            Specific Herbie model product string.
        transformations: dict[str, Any], optional
            Optional transformations to apply to the raw data before returning.

        Raises
        ------
        DateRangeError
            If init_time is invalid or forecast_hours are malformed.
        ValueError
            If herbie is not installed.

        Examples
        --------
        >>> reaper = NWPReaper(
        ...     init_time="2026-01-01 00:00",
        ...     forecast_hours=range(1, 19),
        ...     model="hrrr",
        ...     variable="hourly_precip",
        ...     transformations={
        ...         "spatial_subset": {'lat_bounds': (40, 50), 'lon_bounds': (-90, -80)},
        ...         "variable_rename": {"tp": "total_precipitation"},
        ...     }
        ... )
        """
        self.model = model
        self.init_time = init_time
        self.forecast_hours = (
            list(forecast_hours) if isinstance(forecast_hours, range) else forecast_hours
        )
        if search_str is not None:
            self.search_str = search_str
        elif variable and model in NWP_SEARCH_STRINGS and variable in NWP_SEARCH_STRINGS[model]:
            self.search_str = NWP_SEARCH_STRINGS[model][variable]
        else:
            raise ValueError(
                f"Invalid variable '{variable}' for model '{model}'. "
                f"Available variables: {list(NWP_SEARCH_STRINGS.get(model, {}).keys())}. "
                f"Or provide a custom search_str."
            )
        self.product = product
        self.transformations = transformations

        # Validate parameters
        self._validate_params()

        # Check herbie availability
        if not _HERBIE_AVAILABLE:
            logger.warning(
                "herbie not installed; NWPReaper will not be able to fetch data. "
                "Install with: pip install 'cosecha[nwp]'"
            )
            self._herbie_available = False
        else:
            self._herbie_available = True

        logger.debug(
            f"NWPReaper initialized: model={model}, init_time={init_time}, "
            f"forecast_hours={len(self.forecast_hours) if self.forecast_hours else 'None'}, search_str='{search_str}', product='{product}'"
        )

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
            logger.info(
                f"Fetching HRRR data: model={self.model}, init_time={self.init_time}, "
                f"forecast_hours={self.forecast_hours}"
            )

            # Use default filter (all variables) or a specific filter if desired
            # Users can customize via transformations in sowers
            if self.forecast_hours:
                h = FastHerbie(
                    [self.init_time],
                    model=self.model,
                    fxx=self.forecast_hours,
                    product=self.product,
                )
            else:
                h = FastHerbie([self.init_time], model=self.model, product=self.product)

            ds = h.xarray(search=self.search_str)

            ds.herbie.to_180()  # Convert longitudes to -180 to 180 for easier donwnstream processing

            logger.info(f"Successfully fetched NWP data: {len(ds.data_vars)} variables")
            return ds

        except Exception as e:
            logger.exception("Failed to fetch NWP data")
            raise APIError(f"NWP fetch failed: {e}") from e
        else:
            logger.info(f"Successfully fetched NWP data: {len(ds.data_vars)} variables")
            return ds

    def reap(self) -> HarvestedData:
        """Fetch and return NWP forecast data.

        Returns.
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
        >>> reaper = NWPReaper(
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
                "source_name": f"{self.model} Forecast",
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
                source_name=self.model,
                timestamp=metadata["timestamp"],
                variable_names=variable_names,
                metadata=metadata,
            )

            if self.transformations:
                harvested = apply_gridded_transformations(harvested, self.transformations)

        except APIError:
            raise
        except Exception as e:
            logger.exception("Failed to reap HRRR data")
            raise ReaperError(f"HRRR reaping failed: {e}") from e
        else:
            logger.info(
                f"Successfully reaped HRRR data: {len(variable_names)} variables, "
                f"{len(self.forecast_hours) if self.forecast_hours else 'None'} forecast hours"
            )
            return harvested


# Type hint: NWPReaper implements GriddedReaper protocol
_: type[GriddedReaper] = NWPReaper
