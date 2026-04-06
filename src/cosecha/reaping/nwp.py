"""NWP (Numerical Weather Prediction) reapers for gridded forecast data.

This module implements reapers for high-resolution numerical weather prediction
models (HRRR, RRFS, etc.) using the herbie library for data fetching.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import xarray as xr

from cosecha.exceptions import APIError, DateRangeError, ReaperError
from cosecha.logging import logger
from cosecha.reaping.base import GriddedReaper
from cosecha.utils import apply_gridded_transformations, wrap_errors

__all__ = ["NWPReaper"]


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
}


class NWPReaper(GriddedReaper):
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

    def _check_herbie(self) -> None:
        try:
            import herbie  # noqa: F401,PLC0415
        except ImportError:
            raise ImportError(
                "herbie is not installed. Install with: pip install 'cosecha[nwp]'"
            ) from None

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
        ImportError
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
        super().__init__()
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
        self._check_herbie()

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
        from herbie.fast import FastHerbie  # noqa: PLC0415

        logger.info(
            f"Fetching HRRR data: model={self.model}, init_time={self.init_time}, "
            f"forecast_hours={self.forecast_hours}"
        )

        with wrap_errors(APIError, "NWP fetch failed"):
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
            if not isinstance(ds, xr.Dataset):
                raise APIError("Herbie did not return an xarray Dataset")

            ds.herbie.to_180()
            logger.info(f"Successfully fetched NWP data: {len(ds.data_vars)} variables")
            return ds

    def _reap(self) -> xr.Dataset:
        """Fetch and return NWP forecast data.

        Returns
        -------
        xr.Dataset
            Xarray dataset containing the NWP data.

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
        >>> ds = reaper.reap()
        """
        logger.info(f"Reaping HRRR data for {self.model} model")

        with wrap_errors(ReaperError, "HRRR reaping failed", APIError):
            ds = self._fetch_with_herbie()
            ds = ds.rename({k: k.lower() for k in map(str, ds.dims) if k != k.lower()})

            if self.transformations:
                ds = apply_gridded_transformations(ds, self.transformations)

            logger.info(
                f"Successfully reaped HRRR data: {len(ds.data_vars)} variables, "
                f"{len(self.forecast_hours) if self.forecast_hours else 'None'} forecast hours"
            )
            return ds
