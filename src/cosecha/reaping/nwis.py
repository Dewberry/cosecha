"""USGS NWIS (National Water Information System) data reapers.

This module provides implementations for harvesting hydrological observations
from the USGS NWIS API, including streamflow, stage, and precipitation data.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from typing import Any

import httpx
import pandas as pd

from cosecha.data_models import HarvestedData, validate_date_range
from cosecha.reaping.exceptions import APIError, DateRangeError, InvalidSiteError

__all__ = [
    "USGSStreamflowReaper",
    "USGSStageReaper",
    "USGSPrecipReaper",
]

logger = logging.getLogger(__name__)


class _USGSNWISReaper:
    """Base class for USGS NWIS reapers.

    This class should not be instantiated directly; use subclasses
    (USGSStreamflowReaper, USGSStageReaper, USGSPrecipReaper) instead.

    Parameters
    ----------
    site_ids : list[str]
        List of USGS site IDs (e.g., ["01018035"]).
    start_date : str
        Start date in ISO 8601 format (YYYY-MM-DD).
    end_date : str
        End date in ISO 8601 format (YYYY-MM-DD).
    parameter_code : str
        USGS parameter code (e.g., "00060" for streamflow).
    stat_code : str, optional
        USGS statistic code. By default "00003" (mean daily value).

    Attributes
    ----------
    base_url : str
        Base URL for NWIS API.
    """

    base_url = "https://waterservices.usgs.gov/nwis/dv"

    def __init__(
        self,
        site_ids: list[str],
        start_date: str,
        end_date: str,
        parameter_code: str,
        stat_code: str = "00003",
    ) -> None:
        """Initialize USGS NWIS reaper."""
        self.site_ids = site_ids
        self.start_date = start_date
        self.end_date = end_date
        self.parameter_code = parameter_code
        self.stat_code = stat_code
        self._validate_params()
        logger.debug(
            f"Initialized {self.__class__.__name__}: "
            f"sites={self.site_ids}, dates={self.start_date} to {self.end_date}"
        )

    def _validate_params(self) -> None:
        """Validate initialization parameters.

        Raises
        ------
        InvalidSiteError
            If no site IDs provided.
        DateRangeError
            If dates are invalid.
        """
        if not self.site_ids:
            raise InvalidSiteError("site_ids cannot be empty")

        for site_id in self.site_ids:
            if not isinstance(site_id, str) or not site_id.strip():
                raise InvalidSiteError(f"Invalid site ID: {site_id}")

        try:
            validate_date_range(self.start_date, self.end_date)
        except ValueError as e:
            raise DateRangeError(str(e))

    def _build_url(self) -> str:
        """Build request URL for NWIS API.

        Returns
        -------
        str
            Complete URL for API request.
        """
        sites = ",".join(self.site_ids)
        params = {
            "sites": sites,
            "startDT": self.start_date,
            "endDT": self.end_date,
            "parameterCd": self.parameter_code,
            "statCd": self.stat_code,
            "siteStatus": "all",
            "format": "json",
        }
        param_str = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{self.base_url}?{param_str}"

    def _fetch_with_retry(self, url: str, max_retries: int = 3) -> dict[str, Any]:
        """Fetch data from URL with exponential backoff retry logic.

        Parameters
        ----------
        url : str
            URL to fetch from.
        max_retries : int, optional
            Maximum number of retry attempts. By default 3.

        Returns
        -------
        dict[str, Any]
            JSON response as dictionary.

        Raises
        ------
        APIError
            If request fails after max_retries attempts.
        """
        for attempt in range(max_retries):
            try:
                logger.debug(f"Fetching {url} (attempt {attempt + 1}/{max_retries})")
                response = httpx.get(url, timeout=30.0)
                response.raise_for_status()
                logger.debug(f"Successfully fetched data from NWIS")
                return response.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (429, 503):
                    # Retryable errors: rate limit or service unavailable
                    if attempt < max_retries - 1:
                        backoff = 2**attempt  # 1, 2, 4 seconds
                        logger.warning(
                            f"API returned {e.response.status_code}, " f"retrying in {backoff}s..."
                        )
                        time.sleep(backoff)
                        continue
                raise APIError(f"NWIS API request failed: {e}")
            except httpx.RequestError as e:
                logger.error(f"Request error: {e}")
                raise APIError(f"Failed to fetch from NWIS: {e}")

        raise APIError(f"Failed after {max_retries} retries")

    def _parse_response(self, response: dict[str, Any]) -> pd.DataFrame:
        """Parse NWIS JSON response into DataFrame.

        Parameters
        ----------
        response : dict[str, Any]
            JSON response from NWIS API.

        Returns
        -------
        pd.DataFrame
            Parsed data with columns: time, site_id, value.

        Raises
        ------
        APIError
            If response parsing fails.
        """
        try:
            if "value" not in response:
                raise APIError("Invalid NWIS response: missing 'value' key")

            time_series = response["value"]["timeSeries"]
            if not time_series:
                raise APIError("No time series data in NWIS response")

            data_rows = []
            for ts in time_series:
                site_id = ts["sourceInfo"]["siteCd"][0]["value"]
                values = ts["values"][0]["value"]

                for val in values:
                    data_rows.append(
                        {
                            "time": pd.to_datetime(val["dateTime"]),
                            "site_id": site_id,
                            "value": float(val["value"]) if val["value"] else None,
                        }
                    )

            if not data_rows:
                logger.warning("NWIS response contained no data values")
                return pd.DataFrame(columns=["time", "site_id", "value"])

            df = pd.DataFrame(data_rows)
            df = df.sort_values("time").reset_index(drop=True)
            logger.debug(f"Parsed {len(df)} records from NWIS response")
            return df

        except (KeyError, ValueError, TypeError) as e:
            raise APIError(f"Failed to parse NWIS response: {e}")

    def reap(self) -> HarvestedData:
        """Fetch data from NWIS and return as HarvestedData.

        Returns
        -------
        HarvestedData
            Harvested data with DataFrame and metadata.

        Raises
        ------
        APIError
            If fetching or parsing fails.
        """
        url = self._build_url()
        logger.info(f"Reaping data from NWIS for {len(self.site_ids)} site(s)")

        response = self._fetch_with_retry(url)
        data = self._parse_response(response)

        metadata = {
            "sites": self.site_ids,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "parameter_code": self.parameter_code,
            "stat_code": self.stat_code,
            "record_count": len(data),
        }

        harvested = HarvestedData(
            data=data,
            source_name="USGS_NWIS",
            timestamp=datetime.now(UTC),
            variable_names=[self._get_variable_name()],
            metadata=metadata,
        )

        logger.info(f"Successfully reaped {len(data)} records from {len(self.site_ids)} site(s)")
        return harvested

    def _get_variable_name(self) -> str:
        """Return variable name based on parameter code.

        Returns
        -------
        str
            Human-readable variable name.
        """
        raise NotImplementedError


class USGSStreamflowReaper(_USGSNWISReaper):
    """Reaper for USGS streamflow data.

    Harvests daily mean streamflow measurements from NWIS.

    Parameters
    ----------
    site_ids : list[str]
        List of USGS site IDs.
    start_date : str
        Start date in ISO 8601 format (YYYY-MM-DD).
    end_date : str
        End date in ISO 8601 format (YYYY-MM-DD).

    Examples
    --------
    >>> reaper = USGSStreamflowReaper(
    ...     site_ids=["01018035"],
    ...     start_date="2026-01-01",
    ...     end_date="2026-01-31",
    ... )
    >>> data = reaper.reap()
    >>> data.is_timeseries()
    True
    """

    def __init__(
        self,
        site_ids: list[str],
        start_date: str,
        end_date: str,
    ) -> None:
        """Initialize streamflow reaper with parameter code 00060."""
        super().__init__(
            site_ids=site_ids,
            start_date=start_date,
            end_date=end_date,
            parameter_code="00060",  # Discharge, cubic feet per second
            stat_code="00003",  # Mean
        )

    def _get_variable_name(self) -> str:
        """Return 'streamflow' as variable name."""
        return "streamflow"


class USGSStageReaper(_USGSNWISReaper):
    """Reaper for USGS stage (water level) data.

    Harvests daily mean stage measurements from NWIS.

    Parameters
    ----------
    site_ids : list[str]
        List of USGS site IDs.
    start_date : str
        Start date in ISO 8601 format (YYYY-MM-DD).
    end_date : str
        End date in ISO 8601 format (YYYY-MM-DD).

    Examples
    --------
    >>> reaper = USGSStageReaper(
    ...     site_ids=["01018035"],
    ...     start_date="2026-01-01",
    ...     end_date="2026-01-31",
    ... )
    >>> data = reaper.reap()
    """

    def __init__(
        self,
        site_ids: list[str],
        start_date: str,
        end_date: str,
    ) -> None:
        """Initialize stage reaper with parameter code 00065."""
        super().__init__(
            site_ids=site_ids,
            start_date=start_date,
            end_date=end_date,
            parameter_code="00065",  # Gage height, feet
            stat_code="00003",  # Mean
        )

    def _get_variable_name(self) -> str:
        """Return 'stage' as variable name."""
        return "stage"


class USGSPrecipReaper(_USGSNWISReaper):
    """Reaper for USGS precipitation data.

    Harvests daily accumulated precipitation from NWIS.

    Parameters
    ----------
    site_ids : list[str]
        List of USGS site IDs.
    start_date : str
        Start date in ISO 8601 format (YYYY-MM-DD).
    end_date : str
        End date in ISO 8601 format (YYYY-MM-DD).

    Examples
    --------
    >>> reaper = USGSPrecipReaper(
    ...     site_ids=["01018035"],
    ...     start_date="2026-01-01",
    ...     end_date="2026-01-31",
    ... )
    >>> data = reaper.reap()
    """

    def __init__(
        self,
        site_ids: list[str],
        start_date: str,
        end_date: str,
    ) -> None:
        """Initialize precipitation reaper with parameter code 00045."""
        super().__init__(
            site_ids=site_ids,
            start_date=start_date,
            end_date=end_date,
            parameter_code="00045",  # Precipitation, inches
            stat_code="00006",  # Sum
        )

    def _get_variable_name(self) -> str:
        """Return 'precipitation' as variable name."""
        return "precipitation"
