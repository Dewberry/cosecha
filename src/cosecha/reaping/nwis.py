"""USGS NWIS (National Water Information System) data reapers.

This module provides implementations for harvesting hydrological observations
from the USGS NWIS API, including streamflow, stage, and precipitation data.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd
from dataretrieval import nwis as dr_nwis

from cosecha.data_models import HarvestedData, validate_date_range
from cosecha.logging_config import get_logger
from cosecha.reaping.exceptions import APIError, DateRangeError, InvalidSiteError
from cosecha.reaping.utils import apply_ts_transformations

__all__ = [
    "USGSNWISReaper",
]

logger = get_logger(__name__)

_PARAM_MAPPING = {
    "00060": {"data_type": "instantaneous streamflow", "variable_name": "streamflow"},
    "00065": {"data_type": "instantaneous stage", "variable_name": "stage"},
    "00045": {"data_type": "instantaneous precipitation", "variable_name": "precipitation"},
}

class USGSNWISReaper:
    """Reaper for USGS NWIS instantaneous data."""

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
            raise DateRangeError(str(e)) from e

    def __init__(
        self,
        site_ids: list[str],
        start_date: str,
        end_date: str,
        parameter_code: str | None = None,
        transformations: dict[str, Any] | None = None,
    ) -> None:
        """ Uses the dataretrieval library to fetch data from USGS NWIS/Water Data APIs.

        Parameters
        ----------
        site_ids : list[str]
            List of USGS site IDs (e.g., ["01018035"]).
        start_date : str
            Start date in ISO 8601 format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ).
        end_date : str
            End date in ISO 8601 format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ).
        parameter_code : str | None, optional
            USGS parameter code (e.g., "00060" for streamflow or "00045" for precipitation). 
            If None, fetches all available parameters.
        transformations : dict[str, Any], optional
            Optional transformations to apply to the data.

        Examples
        --------
        >>> reaper = USGSNWISReaper(
        ...     site_ids=["01018035"],
        ...     start_date="2026-01-01",
        ...     end_date="2026-01-31",
        ...     parameter_code="00060",
        ... )
        >>> data = reaper.reap()
        """
        self.site_ids = site_ids
        self.start_date = start_date
        self.end_date = end_date
        self.transformations = transformations
        self.parameter_code = parameter_code
        self._validate_params()
        logger.debug(
            f"Initialized {self.__class__.__name__}: "
            f"sites={self.site_ids}, dates={self.start_date} to {self.end_date}, "
            f"parameter_code={self.parameter_code}"
        )

    def _get_data_type(self) -> str:
        """Return data type name (e.g., 'instantaneous streamflow', 'instantaneous stage').

        Returns
        -------
        str
            Data type name.
        """
        if self.parameter_code is None:
            return "all instantaneous data"
        return _PARAM_MAPPING.get(self.parameter_code, {}).get("data_type", "instantaneous data")
        
    def _get_variable_name(self) -> str | list[str]:
        """Return variable name based on parameter code.

        Returns
        -------
        str | list[str]
            Human-readable variable name or "multiple parameters" if requested all.
        """
        if self.parameter_code is None:
            return "multiple parameters"
        return _PARAM_MAPPING.get(self.parameter_code, {}).get("variable_name", f"parameter_{self.parameter_code}")

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
            
            if self.parameter_code:
                df, _metadata = dr_nwis.get_iv(
                    sites=sites,
                    start=self.start_date,
                    end=self.end_date,
                    parameterCd=self.parameter_code,
                )
            else:
                df, _metadata = dr_nwis.get_iv(
                    sites=sites,
                    start=self.start_date,
                    end=self.end_date,
                )
        except Exception as e:
            logger.exception("Failed to fetch NWIS data")
            raise APIError(f"Failed to fetch NWIS data: {e}") from e
        else:
            if df.empty:
                logger.warning("NWIS returned no data")
                return pd.DataFrame()

            logger.debug(f"Fetched {len(df)} records from NWIS")
            return df

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
            "parameter_code": self.parameter_code if self.parameter_code else "all",
            "record_count": len(df),
        }

        var_name = self._get_variable_name()
        
        harvested = HarvestedData(
            data=df,
            source_name="USGS_NWIS",
            timestamp=datetime.now(UTC),
            variable_names=[var_name] if isinstance(var_name, str) else list(var_name),
            metadata=metadata,
        )
        if self.transformations:
            harvested = apply_ts_transformations(harvested, self.transformations)

        logger.info(f"Successfully reaped {len(df)} records from {len(self.site_ids)} site(s)")
        return harvested
