"""USACE CDA (Corps Data Access) reservoir time series data reapers.

This module provides implementations for harvesting reservoir data from the
USACE CDA Reporting API, including storage, elevation, and outflow data.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import requests

from cosecha._logging import logger
from cosecha._utils import apply_ts_transformations, wrap_errors
from cosecha.exceptions import APIError, DateRangeError, InvalidSiteError
from cosecha.reaping.base import TimeSeriesReaper

__all__ = [
    "ReservoirReaper",
]

BASE_URL = "https://water.usace.army.mil/cda/reporting/providers/{provider}/timeseries"

PARAMETERS = {
    "storage": "{code}.Stor.Inst.1Hour.0.Decodes-Rev",
    "elevation": "{code}.Elev.Inst.1Hour.0.Decodes-Rev",
    "outflow": "{code}-Gated_Total.Flow-Out.Inst.1Hour.0.Rev-SWF-REGI",
    "inflow": "{code}.Flow-In.Ave.~1Day.1Day.Computed-SWF-REGI",  # daily only
}


class ReservoirReaper(TimeSeriesReaper):
    """Reaper for USACE CDA reservoir time series data."""

    def _validate_params(self) -> None:
        """Validate initialization parameters.

        Raises
        ------
        InvalidSiteError
            If no site IDs provided or any are invalid.
        DateRangeError
            If dates are invalid.
        """
        if not self.site_ids:
            raise InvalidSiteError("site_ids cannot be empty")

        for site_id in self.site_ids:
            if not isinstance(site_id, str) or not site_id.strip():
                raise InvalidSiteError(f"Invalid site ID: {site_id}")

        if not self.params:
            raise InvalidSiteError("params cannot be empty")

        for param in self.params:
            if param not in PARAMETERS:
                raise InvalidSiteError(
                    f"Unknown parameter: {param!r}. "
                    f"Available: {list(PARAMETERS.keys())}"
                )

        if self.start_date > self.end_date:
            raise DateRangeError(
                f"start_date ({self.start_date}) must be <= end_date ({self.end_date})"
            )

    def __init__(
        self,
        site_ids: list[str],
        params: list[str],
        start_date: str,
        end_date: str,
        provider: str = "swf",
        transformations: dict[str, Any] | None = None,
    ) -> None:
        """Fetch reservoir time series data from the USACE CDA Reporting API.

        Parameters
        ----------
        site_ids : list[str]
            List of USACE site codes (e.g., ["JPLT2", "BNBT2"]).
        params : list[str]
            List of parameter names to fetch. Available: "storage", "elevation", "outflow", "inflow".
        start_date : str
            Start time in ISO 8601 format (e.g., "2026-06-04T00:00:00Z").
        end_date : str
            End time in ISO 8601 format (e.g., "2026-06-05T00:00:00Z").
        provider : str, optional
            USACE district code (default: "swf" for Fort Worth).
        transformations : dict[str, Any], optional
            Optional transformations to apply to the data.

        Examples
        --------
        >>> reaper = ReservoirReaper(
        ...     site_ids=["JPLT2", "BNBT2"],
        ...     params=["storage", "elevation", "outflow", "inflow"],
        ...     start_date="2026-06-04T00:00:00Z",
        ...     end_date="2026-06-05T00:00:00Z",
        ... )
        >>> data = reaper.reap()
        >>> reaper.sow_to_netcdf("usace_reservoirs.nc")
        """
        super().__init__()
        self.site_ids = site_ids
        self.params = params
        self.provider = provider
        self.transformations = transformations
        try:
            self.start_date = pd.to_datetime(start_date)
            self.end_date = pd.to_datetime(end_date)
        except Exception as e:
            raise DateRangeError(f"Could not parse date: {e}") from e
        self._validate_params()
        logger.debug(
            f"Initialized {self.__class__.__name__}: "
            f"sites={self.site_ids}, params={self.params}, "
            f"dates={self.start_date} to {self.end_date}, provider={self.provider}"
        )

    def _fetch_timeseries(self, ts_name: str) -> tuple[pd.DataFrame, dict]:
        """Fetch a single time series from the USACE CDA reporting API.

        Parameters
        ----------
        ts_name : str
            Full time series identifier.

        Returns
        -------
        tuple[pd.DataFrame, dict]
            DataFrame with datetime and value columns, and response metadata.

        Raises
        ------
        APIError
            If the request fails.
        """
        url = BASE_URL.format(provider=self.provider)
        request_params = {
            "name": ts_name,
            "begin": self.start_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": self.end_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

        with wrap_errors(APIError, f"Failed to fetch USACE time series: {ts_name}"):
            resp = requests.get(url, params=request_params, timeout=60)
            resp.raise_for_status()
            data = resp.json()

        if not data.get("values"):
            return pd.DataFrame(), data

        df = pd.DataFrame(data["values"], columns=["datetime", "value"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df, data

    def _fetch_and_parse(self) -> pd.DataFrame:
        """Fetch all requested parameters for all sites.

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame with columns: site_id, datetime, variable, value, unit.

        Raises
        ------
        APIError
            If fetching fails.
        """
        logger.debug(
            f"Fetching {len(self.params)} parameter(s) for "
            f"{len(self.site_ids)} site(s) from {self.start_date} to {self.end_date}"
        )

        records: list[pd.DataFrame] = []

        for code in self.site_ids:
            for param in self.params:
                ts_name = PARAMETERS[param].format(code=code)
                logger.debug(f"Fetching {code} | {param}: {ts_name}")

                df, meta = self._fetch_timeseries(ts_name)
                if df.empty:
                    logger.warning(f"No data returned for {code} | {param}")
                    continue

                df["site_id"] = code
                df["variable"] = param
                df["unit"] = meta.get("unit", "")
                records.append(df)
                logger.debug(f"Fetched {len(df)} records for {code} | {param}")

        if not records:
            logger.warning("USACE returned no data for any site/parameter combination")
            return pd.DataFrame()

        return pd.concat(records, ignore_index=True)

    def _reap(self) -> pd.DataFrame:
        """Fetch data from USACE CDA and return as a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame with columns: site_id, datetime, variable, value, unit.

        Raises
        ------
        APIError
            If fetching fails.
        """
        logger.info(
            f"Reaping data from USACE CDA for {len(self.site_ids)} site(s), "
            f"{len(self.params)} parameter(s)"
        )

        df = self._fetch_and_parse()

        if self.transformations:
            df = apply_ts_transformations(df, self.transformations)

        logger.info(
            f"Successfully reaped {len(df)} records from "
            f"{len(self.site_ids)} site(s)"
        )
        return df
