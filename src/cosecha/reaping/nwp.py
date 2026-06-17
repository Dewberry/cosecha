"""NWP (Numerical Weather Prediction) reapers for gridded forecast data.

This module implements reapers for high-resolution numerical weather prediction
models (HRRR, RRFS, etc.) using the herbie library for data fetching.
"""

from __future__ import annotations

import warnings
from typing import Any

import pandas as pd
import xarray as xr

from cosecha._logging import logger
from cosecha._utils import apply_gridded_transformations, to_180, wrap_errors
from cosecha.exceptions import APIError, DateRangeError, ReaperError
from cosecha.reaping.base import GriddedReaper

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
            If forecast_hours are invalid.
        """
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

    def _get_latest_model_init(self) -> pd.Timestamp:
        from herbie import HerbieLatest  # noqa: PLC0415

        with wrap_errors(APIError, "Could not fetch latest model initialization time from Herbie"):
            h = HerbieLatest(model=self.model, fxx=max(self.forecast_hours))
            return h.date

    def __init__(
        self,
        init_time: str,
        forecast_hours: list[int] | range | None = None,
        model: str = "hrrr",
        variable: str | list[str] | None = "hourly_precip",
        search_str: str | list[str] | None = None,
        product: str | None = None,
        transformations: dict[str, Any] | None = None,
    ) -> None:
        """Initialize NWPReaper.

        Parameters
        ----------
        init_time : str
            Model initialization time in format "YYYY-MM-DD HH:MM" or similar.
            Parsed by ``pandas.to_datetime()``. Also accepts "latest" to automatically fetch
            the most recent initialization time for the specified model.
        forecast_hours : list[int] | range | None, optional
            Forecast hours to request (e.g., [1, 6, 12] or range(1, 19)). Can be none if fetching analysis product.
        model : str, optional
            NWP model name (default: 'hrrr'). Other options: 'rrfs', 'rtma', etc.
        variable : str | list[str] | None, optional
            A simplified variable name (or list of names) mapping to predefined GRIB regex search
            strings. Common examples include 'hourly_precip', 'total_precip',
            'temp_2m'.
        search_str : str | list[str] | None, optional
            Exact GRIB regex search string(s) to use. Can be combined with ``variable``.
        product : str | None, optional
            Specific Herbie model product string.
        transformations : dict[str, Any] | None, optional
            Optional transformations to apply to the raw data before returning.

        Raises
        ------
        ValueError
            If init_time is invalid or forecast_hours are malformed.
        ReaperError
            If ``variable`` is not recognized for the given ``model``,
            or neither ``variable`` nor ``search_str`` are provided.
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
        self._check_herbie()
        self.model = model
        self.forecast_hours = (
            list(forecast_hours) if isinstance(forecast_hours, range) else forecast_hours
        )

        if init_time == "latest":
            self.init_time = self._get_latest_model_init()
        else:
            self.init_time = pd.to_datetime(init_time)

        search_parts = []
        if search_str is not None:
            if isinstance(search_str, str):
                search_parts.append(search_str)
            else:
                search_parts.extend(search_str)

        if variable is not None:
            variables = [variable] if isinstance(variable, str) else variable
            for var in variables:
                if model in NWP_SEARCH_STRINGS and var in NWP_SEARCH_STRINGS[model]:
                    search_parts.append(NWP_SEARCH_STRINGS[model][var])
                else:
                    raise ReaperError(
                        f"Invalid variable '{var}' for model '{model}'. "
                        f"Available variables: {list(NWP_SEARCH_STRINGS.get(model, {}).keys())}."
                    )

        if not search_parts:
            raise ReaperError("Must provide at least one variable or search_str.")

        self.search_str = f"(?:{'|'.join(search_parts)})"

        self.product = product
        self.transformations = transformations

        self._validate_params()

        logger.debug(
            f"NWPReaper initialized: model={self.model}, init_time={self.init_time}, "
            f"forecast_hours={len(self.forecast_hours) if self.forecast_hours else 'None'}, "
            f"search_str='{self.search_str}', product='{self.product}'"
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
            f"Fetching {self.model.upper()} data: init_time={self.init_time}, "
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

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="In a future version of xarray",
                    category=FutureWarning,
                )
                ds = h.xarray(search=self.search_str)

            if isinstance(ds, list):
                ds = xr.merge(ds, compat="override")

            if not isinstance(ds, xr.Dataset):
                raise APIError("Herbie did not return an xarray Dataset")

            ds = to_180(ds)
            logger.info(
                f"Successfully fetched {self.model.upper()} data: {len(ds.data_vars)} variables"
            )
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
        logger.info(f"Reaping {self.model.upper()} data")

        with wrap_errors(ReaperError, f"{self.model.upper()} reaping failed", APIError):
            ds = self._fetch_with_herbie()
            ds = ds.rename({k: k.lower() for k in map(str, ds.dims) if k != k.lower()})

            if self.transformations:
                ds = apply_gridded_transformations(ds, self.transformations)

            logger.info(
                f"Successfully reaped {self.model.upper()} data: {len(ds.data_vars)} variables, "
                f"{len(self.forecast_hours) if self.forecast_hours else 'None'} forecast hours"
            )
            return ds
