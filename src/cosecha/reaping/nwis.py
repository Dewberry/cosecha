"""USGS NWIS (National Water Information System) data reapers.

This module provides implementations for harvesting hydrological observations
from the USGS NWIS API, including streamflow, stage, and precipitation data.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from dataretrieval import waterdata as dr_waterdata

from cosecha._logging import logger
from cosecha._utils import apply_ts_transformations, parse_date_range, wrap_errors
from cosecha.exceptions import APIError, InvalidSiteError
from cosecha.reaping.base import TimeSeriesReaper

__all__ = [
    "USGSNWISReaper",
]


class USGSNWISReaper(TimeSeriesReaper):
    """Reaper for USGS NWIS instantaneous data."""

    def _validate_params(self) -> None:
        """Validate initialization parameters.

        Raises
        ------
        InvalidSiteError
            If no site IDs provided.
        """
        if not self.site_ids:
            raise InvalidSiteError("site_ids cannot be empty")

        for site_id in self.site_ids:
            if not isinstance(site_id, str) or not site_id.strip():
                raise InvalidSiteError(f"Invalid site ID: {site_id}")

    def __init__(
        self,
        site_ids: list[str],
        start_date: str,
        end_date: str,
        parameter_code: str | list[str] | None = None,
        transformations: dict[str, Any] | None = None,
    ) -> None:
        """Use the dataretrieval library to fetch data from USGS NWIS/Water Data APIs.

        Parameters
        ----------
        site_ids : list[str]
            List of USGS site IDs (e.g., ["01018035"]).
        start_date : str
            Start date in ISO 8601 format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ).
        end_date : str
            End date in ISO 8601 format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ).
        parameter_code : str | list[str] | None, optional
            USGS parameter code (e.g., "00060" for streamflow or "00045" for precipitation).
            If a list is provided, fetches data for all specified parameters.
            If None, fetches all available parameters.
        transformations : dict[str, Any], optional
            Optional transformations to apply to the data.

        Examples
        --------
        >>> reaper = USGSNWISReaper(
        ...     site_ids=["01018035"],
        ...     start_date="2026-01-01",
        ...     end_date="2026-01-31",
        ...     parameter_code=["00060", "00045"],
        ... )
        >>> data = reaper.reap()
        """
        super().__init__()
        self.site_ids = site_ids
        self.start_date, self.end_date = parse_date_range(start_date, end_date)
        self.transformations = transformations
        self.parameter_code = parameter_code
        self._validate_params()
        logger.debug(
            f"Initialized {self.__class__.__name__}: "
            f"sites={self.site_ids}, dates={self.start_date} to {self.end_date}, "
            f"parameter_code={self.parameter_code}"
        )

    def _fetch_and_parse(self) -> pd.DataFrame:
        """Fetch data from NWIS using dataretrieval and parse into DataFrame.

        Returns
        -------
        pd.DataFrame
            Parsed data with site_no, datetime index, and parameter columns.

        Raises
        ------
        APIError
            If fetching fails.
        """
        logger.debug(
            f"Fetching data for "
            f"{len(self.site_ids)} site(s) from {self.start_date} to {self.end_date}"
        )

        sites = [f"USGS-{s}" if not s.startswith("USGS-") else s for s in self.site_ids]
        time_range = f"{self.start_date.isoformat()}/{self.end_date.isoformat()}"

        with wrap_errors(APIError, "Failed to fetch NWIS data"):
            df, _metadata = dr_waterdata.get_continuous(
                monitoring_location_id=sites,
                time=time_range,
                parameter_code=self.parameter_code,
            )

            if df.empty:
                logger.warning("NWIS returned no data")
                return pd.DataFrame()

            logger.debug(f"Fetched {len(df)} records from NWIS")
            return df

    def _reap(self) -> pd.DataFrame:
        """Fetch data from NWIS and return as a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with site_no, datetime index, parameter columns.

        Raises
        ------
        APIError
            If fetching fails.
        """
        logger.info(f"Reaping data from NWIS for {len(self.site_ids)} site(s)")

        df = self._fetch_and_parse()

        if self.transformations:
            df = apply_ts_transformations(df, self.transformations)

        logger.info(f"Successfully reaped {len(df)} records from {len(self.site_ids)} site(s)")
        return df
