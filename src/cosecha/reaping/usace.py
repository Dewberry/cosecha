"""USACE CDA (Corps Data Access) reservoir time series data reapers.

This module provides implementations for harvesting reservoir data from the
USACE CDA Reporting API, including storage, elevation, and outflow data.
"""

from __future__ import annotations

import contextlib
from typing import Any

import pandas as pd
import tiny_retriever

from cosecha._logging import logger
from cosecha._utils import apply_ts_transformations, wrap_errors
from cosecha.exceptions import APIError, DateRangeError, InvalidSiteError
from cosecha.reaping.base import TimeSeriesReaper

__all__ = [
    "ReservoirReaper",
]

BASE_URL = "https://water.usace.army.mil/cda/reporting/providers/{provider}/timeseries"
LOCATIONS_URL = "https://water.usace.army.mil/cda/reporting/providers/{provider}/locations"

LABEL_MAP = {
    "storage": "Flood Storage",
    "elevation": "Elevation",
    "outflow": "Outflow",
    "inflow": "Inflow",
}


class ReservoirReaper(TimeSeriesReaper):
    """Reaper for USACE CDA reservoir time series data."""

    timeout: int = 120
    catalog_timeout: int = 60

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
            if param not in LABEL_MAP:
                raise InvalidSiteError(
                    f"Unknown parameter: {param!r}. Available: {list(LABEL_MAP)}"
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

    def _get_catalog(self) -> list[dict[str, Any]]:
        """Fetch the locations catalog for this provider."""
        url = LOCATIONS_URL.format(provider=self.provider)
        with wrap_errors(APIError, "Failed to fetch USACE locations catalog"):
            return tiny_retriever.fetch(url, "json", timeout=self.catalog_timeout)

    def _build_urls(self) -> tuple[list[str], list[str], list[str]]:
        """Discover tsids from the catalog and build URLs for each site/param."""
        catalog = self._get_catalog()

        # Build a lookup: site code to list of timeseries
        site_lookup = {
            location.get("code", "").upper(): location.get("timeseries") or []
            for location in catalog
        }

        # For each site and param, find the tsid and build the URL
        base = BASE_URL.format(provider=self.provider)
        begin = self.start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        end = self.end_date.strftime("%Y-%m-%dT%H:%M:%SZ")

        codes = []
        params = []
        urls = []

        for site_id in self.site_ids:
            timeseries_list = site_lookup.get(site_id.upper(), [])
            if not timeseries_list:
                logger.warning(f"Site {site_id} not found in {self.provider} catalog")
                continue

            for param in self.params:
                label = LABEL_MAP[param]
                tsid = next(
                    (ts.get("tsid") for ts in timeseries_list if ts.get("label") == label), None
                )
                if tsid:
                    codes.append(site_id)
                    params.append(param)
                    urls.append(f"{base}?name={tsid}&begin={begin}&end={end}")
                else:
                    logger.warning(f"No '{LABEL_MAP[param]}' timeseries found for {site_id}")

        return codes, params, urls

    def _fetch(self, urls: list[str]) -> list[dict[str, Any]]:
        """Fetch all URLs asynchronously via tiny_retriever."""
        with wrap_errors(APIError, "Failed to fetch USACE time series"):
            return tiny_retriever.fetch(urls, "json", timeout=self.timeout)

    def _parse_single(self, code: str, param: str, data: dict[str, Any]) -> pd.DataFrame | None:
        """Parse a single JSON response into a DataFrame, or None if empty."""
        if not data or not data.get("values"):
            logger.warning(f"No data returned for {code} | {param}")
            return None
        df = pd.DataFrame(data["values"], columns=["datetime", "value"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["site_id"] = code
        df["variable"] = param
        df["unit"] = data.get("unit", "")
        return df

    def _parse_responses(
        self,
        codes: list[str],
        params: list[str],
        responses: list[dict[str, Any]],
    ) -> pd.DataFrame:
        """Parse JSON responses into a long-format DataFrame."""
        records = (
            df
            for c, p, data in zip(codes, params, responses, strict=True)
            if (df := self._parse_single(c, p, data)) is not None
        )
        with contextlib.suppress(ValueError):
            return pd.concat(records, ignore_index=True)
        logger.warning("USACE returned no data for any site/parameter combination")
        return pd.DataFrame()

    def _reap(self) -> pd.DataFrame:
        """Fetch data from USACE CDA and return as a pandas DataFrame."""
        logger.info(
            f"Reaping USACE CDA: {len(self.site_ids)} site(s), {len(self.params)} parameter(s)"
        )

        codes, params, urls = self._build_urls()
        if not urls:
            logger.warning("No time series found for any site/parameter combination")
            return pd.DataFrame()

        responses = self._fetch(urls)
        df = self._parse_responses(codes, params, responses)

        if self.transformations:
            df = apply_ts_transformations(df, self.transformations)

        logger.info(f"Reaped {len(df)} records from {len(self.site_ids)} site(s)")
        return df
