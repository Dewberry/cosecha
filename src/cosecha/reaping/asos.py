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
from cosecha.exceptions import APIError, DataNotFoundError, DateRangeError, InvalidSiteError
from cosecha.reaping.base import TimeSeriesReaper

__all__ = [
    "ASOSReaper",
]

BASE_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
_IEM_ALL_VARS = "all"


class ASOSReaper(TimeSeriesReaper):
    """Reaper for IEM ASOS data."""

    timeout: int = 120

    def _validate_params(self) -> None:
        """Validate initialization parameters.

        Raises
        ------
        InvalidSiteError
            If no state is provided.
        DateRangeError
            If dates are invalid.
        """
        if not self.state:
            raise InvalidSiteError("state cannot be empty")

        if self.start_date > self.end_date:
            raise DateRangeError(
                f"start_date ({self.start_date}) must be <= end_date ({self.end_date})"
            )

    def __init__(
        self,
        start_date: str,
        end_date: str,
        state: str = "ASOS",
        variable: str | list[str] | None = None,
        transformations: dict[str, Any] | None = None,
    ) -> None:
        """Fetch data from IEM ASOS API.

        Parameters
        ----------
        start_date : str
            Start date in ISO 8601 format (YYYY-MM-DD).
        end_date : str
            End date in ISO 8601 format (YYYY-MM-DD).
        state : str, optional
            State abbreviation (e.g., 'TX' or 'ASOS' for all), by default "ASOS".
        variable : str | list[str] | None, optional
            Variable to fetch (e.g., 'p01i', 'tmpf', or ['p01i', 'tmpf']).
            If None, fetches 'all'.
        transformations : dict[str, Any], optional
            Optional transformations to apply to the data.
        """
        super().__init__()
        self.state = state
        self.network = "ASOS" if state.upper() == "ASOS" else f"{state.upper()}_ASOS"

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

        self._validate_params()
        logger.debug(
            f"Initialized {self.__class__.__name__}: "
            f"network={self.network}, dates={self.start_date} to {self.end_date}, "
            f"data={self.data_vars}"
        )

    def _build_url(self) -> str:
        """Build the IEM ASOS request URL with query parameters."""
        params = [
            ("network", self.network),
            ("year1", self.start_date.year),
            ("month1", self.start_date.month),
            ("day1", self.start_date.day),
            ("year2", self.end_date.year),
            ("month2", self.end_date.month),
            ("day2", self.end_date.day),
            ("format", "comma"),
            ("latlon", "yes"),
        ]
        for var in self.data_vars:
            params.append(("data", var))
        return f"{BASE_URL}?{urlencode(params)}"

    def _fetch(self, url: str) -> str:
        """Fetch CSV text from IEM via tiny_retriever."""
        with wrap_errors(APIError, f"Failed to fetch ASOS data for {self.network}"):
            return tiny_retriever.fetch(url, "text", timeout=self.timeout)

    def _parse_response(self, text: str) -> pd.DataFrame:
        """Parse IEM CSV text into a DataFrame.

        The first 5 rows of IEM ASOS output are comments (skiprows=5).
        """
        df = pd.read_csv(StringIO(text), skiprows=5)
        if df.empty:
            raise DataNotFoundError(
                f"ASOS returned no data for network {self.network} and time range {self.start_date} to {self.end_date}"
            )
        logger.debug(f"Fetched {len(df)} records from ASOS for {self.network}")
        return df

    def _reap(self) -> pd.DataFrame:
        """Fetch data from ASOS and return as a pandas DataFrame."""
        logger.info(
            f"Reaping ASOS data: network={self.network}, "
            f"data={self.data_vars}"
        )

        url = self._build_url()
        text = self._fetch(url)
        df = self._parse_response(text)

        if self.transformations and not df.empty:
            df = apply_ts_transformations(df, self.transformations)

        logger.info(f"Reaped {len(df)} records from {self.network}")
        return df
