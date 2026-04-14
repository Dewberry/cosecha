"""IEM ASOS data reapers.

This module provides implementations for harvesting ASOS observations
from the Iowa Environmental Mesonet (IEM) API.
"""

from __future__ import annotations

from io import StringIO
from typing import Any

import pandas as pd
import requests
import time

from cosecha._logging import logger
from cosecha._utils import apply_ts_transformations, wrap_errors
from cosecha.exceptions import APIError, DateRangeError, InvalidSiteError
from cosecha.reaping.base import TimeSeriesReaper

__all__ = [
    "ASOSReaper",
]


class ASOSReaper(TimeSeriesReaper):
    """Reaper for IEM ASOS data."""

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
        data: str | list[str] | None = None,
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
        data : str | list[str] | None, optional
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

        if data is None:
            self.data_vars = ["all"]
        elif isinstance(data, str):
            self.data_vars = [data]
        else:
            self.data_vars = data

        self.transformations = transformations

        self._validate_params()
        logger.debug(
            f"Initialized {self.__class__.__name__}: "
            f"network={self.network}, dates={self.start_date} to {self.end_date}, "
            f"data={self.data_vars}"
        )

    def _fetch_and_parse(self) -> pd.DataFrame:
        """Fetch data from IEM and parse into DataFrame.

        Returns
        -------
        pd.DataFrame
            Parsed data from ASOS CSV output.

        Raises
        ------
        APIError
            If fetching fails.
        """
        url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

        params: dict[str, Any] = {
            "network": self.network,
            "data": self.data_vars,
            "year1": self.start_date.year,
            "month1": self.start_date.month,
            "day1": self.start_date.day,
            "year2": self.end_date.year,
            "month2": self.end_date.month,
            "day2": self.end_date.day,
            "format": "comma",
            "latlon": "yes",
        }

        logger.debug(
            f"Fetching ASOS data for network {self.network} "
            f"from {self.start_date} to {self.end_date}"
        )

        with wrap_errors(APIError, f"Failed to fetch ASOS data for {self.network}"):
            response: requests.Response | None = None
            for attempt in range(3):
                try:
                    response = requests.get(url, params=params)
                    response.raise_for_status()
                    break
                except requests.RequestException as e:
                    if attempt == 2:
                        raise e
                    logger.warning(f"Fetch failed (attempt {attempt + 1}/3): {e}. Retrying in 5 seconds...")
                    time.sleep(5)

            if response is None:
                raise APIError(f"Failed to fetch ASOS data for {self.network}: No response")

            # The first 5 rows of IEM ASOS output are comments (skiprows=5)
            df = pd.read_csv(StringIO(response.text), skiprows=5)

            if df.empty:
                logger.warning(f"ASOS returned no data for network {self.network}")
                return pd.DataFrame()

            logger.debug(f"Fetched {len(df)} records from ASOS for {self.network}")
            return df

    def _reap(self) -> pd.DataFrame:
        """Fetch data from ASOS and return as a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with ASOS data.

        Raises
        ------
        APIError
            If fetching fails.
        """
        logger.info(f"Reaping ASOS data from network {self.network}")

        df = self._fetch_and_parse()

        if self.transformations and not df.empty:
            df = apply_ts_transformations(df, self.transformations)

        logger.info(f"Successfully reaped {len(df)} records from {self.network}")
        return df
