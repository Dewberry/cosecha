"""USGS NWIS (National Water Information System) data reapers.

This module provides implementations for harvesting hydrological observations
from the USGS NWIS API, including streamflow, stage, and precipitation data.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from dataretrieval import nwis as dr_nwis

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

    Uses the dataretrieval library to fetch data from USGS NWIS/Water Data APIs.

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
        Not used for instantaneous data.

    """

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
        try:
            logger.debug(
                f"Fetching {self._get_data_type()} data for "
                f"{len(self.site_ids)} site(s) from {self.start_date} to {self.end_date}"
            )

            # Convert site IDs to comma-separated string
            sites = ",".join(self.site_ids)

            # Use appropriate function based on data type
            df, metadata = self._get_data(sites)

            if df.empty:
                logger.warning("NWIS returned no data")
                return pd.DataFrame()

            logger.debug(f"Fetched {len(df)} records from NWIS")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch NWIS data: {e}")
            raise APIError(f"Failed to fetch NWIS data: {e}")

    def _get_data(self, sites: str) -> tuple[pd.DataFrame, Any]:
        """Fetch data from NWIS. To be implemented by subclasses.

        Parameters
        ----------
        sites : str
            Comma-separated site IDs.

        Returns
        -------
        tuple[pd.DataFrame, Any]
            DataFrame and metadata from dataretrieval.
        """
        raise NotImplementedError

    def _get_data_type(self) -> str:
        """Return data type name (e.g., 'streamflow', 'stage'). To be implemented by subclasses.

        Returns
        -------
        str
            Data type name.
        """
        raise NotImplementedError

    def reap(self) -> HarvestedData:
        """Fetch data from NWIS and return as HarvestedData.

        Returns
        -------
        HarvestedData
            Harvested data with DataFrame and metadata.

        Raises
        ------
        APIError
            If fetching fails.
        """
        logger.info(
            f"Reaping {self._get_data_type()} data from NWIS for {len(self.site_ids)} site(s)"
        )

        df = self._fetch_and_parse()

        metadata = {
            "sites": self.site_ids,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "parameter_code": self.parameter_code,
            "record_count": len(df),
        }

        harvested = HarvestedData(
            data=df,
            source_name="USGS_NWIS",
            timestamp=datetime.now(UTC),
            variable_names=[self._get_variable_name()],
            metadata=metadata,
        )

        logger.info(f"Successfully reaped {len(df)} records from {len(self.site_ids)} site(s)")
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

    Harvests instantaneous streamflow measurements from NWIS.

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
            stat_code="00003",  # Mean (not used for instantaneous data)
        )

    def _get_data(self, sites: str) -> tuple[pd.DataFrame, Any]:
        """Fetch instantaneous streamflow data from NWIS.

        Parameters
        ----------
        sites : str
            Comma-separated site IDs.

        Returns
        -------
        tuple[pd.DataFrame, Any]
            DataFrame and metadata from dataretrieval.
        """
        return dr_nwis.get_iv(
            sites=sites,
            start=self.start_date,
            end=self.end_date,
            parameterCd=self.parameter_code,
        )

    def _get_data_type(self) -> str:
        """Return 'instantaneous streamflow' as data type."""
        return "instantaneous streamflow"

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

    def _get_data(self, sites: str) -> tuple[pd.DataFrame, Any]:
        """Fetch instantaneous stage data from NWIS.

        Parameters
        ----------
        sites : str
            Comma-separated site IDs.

        Returns
        -------
        tuple[pd.DataFrame, Any]
            DataFrame and metadata from dataretrieval.
        """
        return dr_nwis.get_iv(
            sites=sites,
            start=self.start_date,
            end=self.end_date,
            parameterCd=self.parameter_code,
        )

    def _get_data_type(self) -> str:
        """Return 'instantaneous stage' as data type."""
        return "instantaneous stage"

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

    def _get_data(self, sites: str) -> tuple[pd.DataFrame, Any]:
        """Fetch instantaneous precipitation data from NWIS.

        Parameters
        ----------
        sites : str
            Comma-separated site IDs.

        Returns
        -------
        tuple[pd.DataFrame, Any]
            DataFrame and metadata from dataretrieval.
        """
        return dr_nwis.get_iv(
            sites=sites,
            start=self.start_date,
            end=self.end_date,
            parameterCd=self.parameter_code,
        )

    def _get_data_type(self) -> str:
        """Return 'instantaneous precipitation' as data type."""
        return "instantaneous precipitation"

    def _get_variable_name(self) -> str:
        """Return 'precipitation' as variable name."""
        return "precipitation"
