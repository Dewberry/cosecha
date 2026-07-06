"""IEM ASOS data reapers.

This module provides implementations for harvesting ASOS observations
from the Iowa Environmental Mesonet (IEM) API.
"""

from __future__ import annotations

from io import StringIO
from typing import Any
from urllib.parse import urlencode

import pandas as pd
import tiny_retriever

from cosecha._logging import logger
from cosecha._utils import apply_ts_transformations, wrap_errors
from cosecha.exceptions import APIError, DataNotFoundError, DateRangeError
from cosecha.reaping.base import TimeSeriesReaper

__all__ = [
    "ASOSReaper",
]

BASE_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
_IEM_ALL_VARS = "all"


class ASOSReaper(TimeSeriesReaper):
    """Reaper for IEM ASOS data."""

    def _validate_params(self) -> None:
        """Validate initialization parameters.

        Raises
        ------
        DateRangeError
            If dates are invalid.
        """
        if self.start_date > self.end_date:
            raise DateRangeError(
                f"start_date ({self.start_date}) must be <= end_date ({self.end_date})"
            )

    def __init__(
        self,
        start_date: str,
        end_date: str,
        state: str | None = None,
        variable: str | list[str] | None = None,
        report_type: int | list[int] | None = None,
        transformations: dict[str, Any] | None = None,
        timeout: int = 120,
    ) -> None:
        """Fetch data from IEM ASOS API.

        Parameters
        ----------
        start_date : str
            Start date in ISO 8601 format (YYYY-MM-DD HH:MMZ).
        end_date : str
            End date in ISO 8601 format (YYYY-MM-DD HH:MMZ).
        state : str | None, optional
            State abbreviation (e.g., 'TX'). If None (default), fetches from
            all networks (IEM limits this to a 24-hour window).
        variable : str | list[str] | None, optional
            Variable to fetch (e.g., 'p01i', 'tmpf', or ['p01i', 'tmpf']).
            If None, fetches 'all'.
        report_type : int | list[int] | None, optional
            IEM report type filter: 1 (HFMETAR/5-min), 3 (routine), 4 (specials).
            If None (default), returns all report types.
        transformations : dict[str, Any], optional
            Optional transformations to apply to the data.
        timeout : int, optional
            Request timeout in seconds, by default 120.
        """
        super().__init__()
        self.state = state
        self.network = f"{state.upper()}_ASOS" if state else None

        try:
            self.start_date = pd.to_datetime(start_date)
            self.end_date = pd.to_datetime(end_date)
        except Exception as e:
            raise DateRangeError(f"Could not parse date: {e}") from e

        if variable is None:
            self.data_vars = [_IEM_ALL_VARS]
        elif isinstance(variable, str):
            self.data_vars = [variable]
        else:
            self.data_vars = variable

        self.transformations = transformations
        self.timeout = timeout

        if report_type is None:
            self.report_type = None
        elif isinstance(report_type, int):
            self.report_type = [report_type]
        else:
            self.report_type = report_type

        self._validate_params()
        logger.debug(
            f"Initialized {self.__class__.__name__}: "
            f"network={self.network}, dates={self.start_date} to {self.end_date}, "
            f"data={self.data_vars}"
        )

    def _build_url(self) -> str:
        """Build the IEM ASOS request URL with query parameters."""
        params = {
            "year1": self.start_date.year,
            "month1": self.start_date.month,
            "day1": self.start_date.day,
            "hour1": self.start_date.hour,
            "minute1": self.start_date.minute,
            "year2": self.end_date.year,
            "month2": self.end_date.month,
            "day2": self.end_date.day,
            "hour2": self.end_date.hour,
            "minute2": self.end_date.minute,
            "tz": "Etc/UTC",
            "format": "comma",
            "latlon": "yes",
            "data": self.data_vars,
        }
        if self.network is not None:
            params["network"] = self.network
        if self.report_type is not None:
            params["report_type"] = self.report_type
        return f"{BASE_URL}?{urlencode(params, doseq=True)}"

    def _fetch(self, url: str) -> str:
        """Fetch CSV text from IEM via tiny_retriever."""
        with wrap_errors(
            APIError, f"Failed to fetch ASOS data for {self.network or 'all networks'}"
        ):
            return tiny_retriever.fetch(url, "text", timeout=self.timeout)

    def _parse_response(self, text: str) -> pd.DataFrame:
        """Parse IEM CSV text into a DataFrame.

        The first 5 rows of IEM ASOS output are comments (skiprows=5).
        """
        df = pd.read_csv(StringIO(text), skiprows=5)
        if df.empty:
            raise DataNotFoundError(
                f"ASOS returned no data for network {self.network or 'all networks'} and time range {self.start_date} to {self.end_date}"
            )
        logger.debug(f"Fetched {len(df)} records from ASOS for {self.network or 'all networks'}")
        return df

    def _reap(self) -> pd.DataFrame:
        """Fetch data from ASOS and return as a pandas DataFrame."""
        logger.info(
            f"Reaping ASOS data: network={self.network or 'all networks'}, data={self.data_vars}"
        )

        url = self._build_url()
        text = self._fetch(url)
        df = self._parse_response(text)

        if self.transformations and not df.empty:
            df = apply_ts_transformations(df, self.transformations)

        logger.info(f"Reaped {len(df)} records from {self.network or 'all networks'}")
        return df
