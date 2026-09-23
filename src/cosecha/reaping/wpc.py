"""WPC (Weather Prediction Center) reaper for gridded quantitative precipitation forecasts.

This module implements a reaper for the NOAA WPC 2.5 km CONUS 6-hour QPF GRIB2
products published at https://ftp-wpc.ncep.noaa.gov/2p5km_qpf/.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any, cast

import pandas as pd
import tiny_retriever
import xarray as xr

from cosecha._logging import logger
from cosecha._utils import apply_gridded_transformations, to_180, wrap_errors
from cosecha.exceptions import APIError, DataNotFoundError, DateRangeError, ReaperError
from cosecha.reaping.base import GriddedReaper

__all__ = [
    "WPCQPFReaper",
]

BASE_URL = "https://ftp-wpc.ncep.noaa.gov/2p5km_qpf/"
_FILE_PATTERN = re.compile(r'href="p06m_(\d{10})f(\d{3})\.grb"')
_STEP_HOURS = 6
# 00Z/12Z issuances extend to 168 hours, 06Z/18Z issuances to 174 hours.
_CYCLE_MAX_HOURS = {0: 168, 6: 174, 12: 168, 18: 174}


class WPCQPFReaper(GriddedReaper):
    """Reaper for NOAA WPC 2.5 km CONUS 6-hour QPF forecasts."""

    def _validate_params(self) -> None:
        """Validate initialization parameters.

        Raises
        ------
        DateRangeError
            If init_time is not a WPC issuance time or forecast_hours are invalid.
        """
        if self.init_time is not None and (
            self.init_time.hour not in _CYCLE_MAX_HOURS
            or self.init_time != self.init_time.floor("h")
        ):
            raise DateRangeError(f"init_time must be at 00, 06, 12 or 18 UTC, got {self.init_time}")

        if self.forecast_hours is not None and not (
            self.forecast_hours
            and all(
                isinstance(h, int)
                and 0 < h <= max(_CYCLE_MAX_HOURS.values())
                and h % _STEP_HOURS == 0
                for h in self.forecast_hours
            )
        ):
            raise DateRangeError(
                f"forecast_hours must be positive multiples of {_STEP_HOURS}, "
                f"got {self.forecast_hours}"
            )

    def __init__(
        self,
        init_time: str,
        forecast_hours: list[int] | range | None = None,
        transformations: dict[str, Any] | None = None,
        timeout: int = 120,
    ) -> None:
        """Initialize WPCQPFReaper.

        Each forecast hour holds the precipitation (variable `tp`, in kg m**-2)
        accumulated over the 6 hours ending at that hour. WPC only keeps roughly the
        last 6 days of issuances online.

        Parameters
        ----------
        init_time : str
            Issuance time at 00, 06, 12 or 18 UTC (e.g. "2026-09-23 12:00Z"). Also accepts
            "latest" to fetch the most recent issuance with all requested forecast hours.
        forecast_hours : list[int] | range | None, optional
            Forecast hours to request (e.g., [6, 12] or range(6, 73, 6)). Must be positive
            multiples of 6. If None, fetches all forecast hours of the issuance.
        transformations : dict[str, Any] | None, optional
            Optional transformations to apply to the raw data before returning.
        timeout : int, optional
            Request timeout in seconds, by default 120.

        Raises
        ------
        DateRangeError
            If init_time or forecast_hours are invalid.

        Examples
        --------
        >>> reaper = WPCQPFReaper(
        ...     init_time="latest",
        ...     forecast_hours=range(6, 73, 6),
        ...     transformations={
        ...         "spatial_subset": {"lat_bounds": (29, 31), "lon_bounds": (-96, -94)},
        ...     },
        ... )
        """
        super().__init__()
        self.is_latest = init_time == "latest"
        self.init_time: pd.Timestamp | None = None
        if not self.is_latest:
            try:
                self.init_time = pd.to_datetime(init_time, utc=True)
            except (ValueError, TypeError) as e:
                raise DateRangeError(f"Could not parse date: {e}") from e

        self.forecast_hours = (
            list(forecast_hours) if isinstance(forecast_hours, range) else forecast_hours
        )
        self.transformations = transformations
        self.timeout = timeout

        self._validate_params()

        logger.debug(
            f"Initialized {self.__class__.__name__}: init_time={init_time}, "
            f"forecast_hours={self.forecast_hours or 'all'}"
        )

    def _requested_hours(self, init_time: pd.Timestamp) -> set[int]:
        """Return the forecast hours to fetch for an issuance."""
        if self.forecast_hours is not None:
            return set(self.forecast_hours)
        return set(range(_STEP_HOURS, _CYCLE_MAX_HOURS[init_time.hour] + 1, _STEP_HOURS))

    def _find_available_files(self) -> list[str]:
        """Find the WPC QPF file URLs to download from the server's directory listing.

        When init_time is "latest", this also sets self.init_time to the most recent
        issuance that has all requested forecast hours.
        """
        with wrap_errors(APIError, f"Could not list available WPC QPF files at {BASE_URL}"):
            listing = tiny_retriever.fetch(BASE_URL, "text", timeout=self.timeout)

        available: dict[pd.Timestamp, set[int]] = {}
        for init_str, hour_str in _FILE_PATTERN.findall(listing):
            init = pd.to_datetime(init_str, format="%Y%m%d%H", utc=True)
            available.setdefault(init, set()).add(int(hour_str))

        if self.is_latest:
            self.init_time = next(
                (
                    init
                    for init in sorted(available, reverse=True)
                    if self._requested_hours(init) <= available[init]
                ),
                None,
            )
            if self.init_time is None:
                raise DataNotFoundError(
                    f"No WPC issuance has all requested forecast hours "
                    f"{self.forecast_hours or 'all'}."
                )
            logger.info(f"Using latest WPC issuance {self.init_time}")

        init_time = cast("pd.Timestamp", self.init_time)
        requested = self._requested_hours(init_time)
        hours = sorted(requested & available.get(init_time, set()))
        if not hours:
            raise DataNotFoundError(
                f"No WPC QPF files found for {init_time}. WPC only keeps roughly the "
                "last 6 days of issuances online."
            )

        missing = sorted(requested - set(hours))
        if missing:
            logger.warning(f"Forecast hours {missing} not available for {init_time}, skipping.")

        return [f"{BASE_URL}p06m_{init_time:%Y%m%d%H}f{hour:03d}.grb" for hour in hours]

    def _process_single_file(self, file: Path) -> xr.Dataset:
        """Load a single WPC QPF GRIB2 file."""
        data_in = xr.load_dataarray(
            file, engine="cfgrib", decode_timedelta=True, backend_kwargs={"indexpath": ""}
        )
        wpc_da = data_in.expand_dims("step")
        wpc_da = wpc_da.assign_coords(valid_time=wpc_da["valid_time"].expand_dims("step"))
        return wpc_da.to_dataset()

    def _fetch_data(self) -> xr.Dataset:
        """Fetch WPC QPF data over HTTPS."""
        urls = self._find_available_files()
        logger.info(f"Fetching {len(urls)} WPC QPF files for {self.init_time}")

        data_arrays: list[xr.Dataset] = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            files = [Path(tmp_dir) / url.rsplit("/", maxsplit=1)[-1] for url in urls]
            with wrap_errors(APIError, "Could not download WPC QPF files"):
                tiny_retriever.download(urls, files, timeout=self.timeout)

            for i, file in enumerate(files):
                wpc_ds = self._process_single_file(file)
                # The lat/lon grid is identical across files, so keep one copy to limit memory.
                data_arrays.append(
                    wpc_ds if i == 0 else wpc_ds.drop_vars(["latitude", "longitude"])
                )

        wpc_ds = xr.concat(data_arrays, dim="step", coords="minimal", compat="override")
        return to_180(wpc_ds)

    def _reap(self) -> xr.Dataset:
        """Fetch and return WPC QPF gridded data.

        Returns
        -------
        xr.Dataset
            WPC QPF gridded dataset.

        Raises
        ------
        ReaperError
            If data fetching fails.
        """
        logger.info(f"Reaping WPC QPF data for {self.init_time or 'latest'}")

        with wrap_errors(ReaperError, "WPC QPF reaping failed"):
            ds = self._fetch_data()
            ds = ds.rename({k: k.lower() for k in map(str, ds.dims) if k != k.lower()})

            if self.transformations:
                ds = apply_gridded_transformations(ds, self.transformations)

            logger.info(f"Successfully reaped WPC QPF data: {ds.sizes.get('step', 0)} steps")
            return ds
