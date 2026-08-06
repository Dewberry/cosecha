"""NWS Local Storm Reports (LSR) reaper.

This module provides an implementation for harvesting NWS Local Storm Reports
from the Iowa Environmental Mesonet (IEM) GeoJSON API at
https://mesonet.agron.iastate.edu/geojson/lsr.php
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import tiny_retriever

from cosecha._logging import logger
from cosecha._utils import apply_ts_transformations, wrap_errors
from cosecha.exceptions import APIError, DataNotFoundError, DateRangeError
from cosecha.reaping.base import TimeSeriesReaper

__all__ = [
    "LSRReaper",
]

BASE_URL = "https://mesonet.agron.iastate.edu/geojson/lsr.php"


class LSRReaper(TimeSeriesReaper):
    """Reaper for NWS Local Storm Reports via the IEM GeoJSON endpoint.

    Fetches storm report features for a given time window, optionally filtering
    by WFO (Weather Forecast Office), event type, and state.
    """

    def _validate_params(self) -> None:
        """Validate initialization parameters.

        Raises
        ------
        DateRangeError
            If dates are invalid.
        """
        if self.start_date >= self.end_date:
            raise DateRangeError(
                f"start_date ({self.start_date}) must be < end_date ({self.end_date})"
            )

    def __init__(
        self,
        start_date: str,
        end_date: str,
        wfos: list[str] | None = None,
        event_types: list[str] | None = None,
        state: str | None = None,
        transformations: dict[str, Any] | None = None,
        timeout: int = 120,
    ) -> None:
        """Fetch NWS Local Storm Reports from the IEM GeoJSON API.

        Parameters
        ----------
        start_date : str
            Start datetime in ISO 8601 format (e.g., "2026-07-01T00:00:00Z").
        end_date : str
            End datetime in ISO 8601 format (e.g., "2026-07-02T00:00:00Z").
        wfos : list[str] | None, optional
            List of Weather Forecast Office codes to query (e.g., ["BOU", "GJT"]).
            If None, fetches reports from all WFOs.
        event_types : list[str] | None, optional
            Event types to include (e.g., ["FLASH FLOOD", "HEAVY RAIN"]).
            If None, returns all event types.
        state : str | None, optional
            Two-letter state abbreviation to filter results (e.g., "CO").
            If None, no state filtering is applied.
        transformations : dict[str, Any], optional
            Optional transformations to apply to the resulting DataFrame.
        timeout : int, optional
            Request timeout in seconds, by default 120.

        Examples
        --------
        >>> reaper = LSRReaper(
        ...     start_date="2026-07-01T00:00:00Z",
        ...     end_date="2026-07-02T00:00:00Z",
        ...     wfos=["BOU", "GJT"],
        ...     event_types=["FLASH FLOOD", "HEAVY RAIN"],
        ...     state="CO",
        ... )
        >>> data = reaper.reap()
        """
        super().__init__()

        try:
            self.start_date = pd.to_datetime(start_date)
            self.end_date = pd.to_datetime(end_date)
        except Exception as e:
            raise DateRangeError(f"Could not parse date: {e}") from e

        self.wfos = [w.upper().strip() for w in wfos] if wfos else None
        self.event_types = [et.upper().strip() for et in event_types] if event_types else None
        self.state = state.upper().strip() if state else None
        self.transformations = transformations
        self.timeout = timeout

        self._validate_params()
        logger.debug(
            f"Initialized {self.__class__.__name__}: "
            f"wfos={self.wfos}, event_types={self.event_types}, state={self.state}, "
            f"dates={self.start_date} to {self.end_date}"
        )

    def _build_url(self) -> str:
        """Build the IEM LSR GeoJSON request URL."""
        sts = self.start_date.strftime("%Y%m%d%H%M")
        ets = self.end_date.strftime("%Y%m%d%H%M")
        url = f"{BASE_URL}?sts={sts}&ets={ets}"
        if self.wfos:
            url += f"&wfos={','.join(self.wfos)}"
        return url

    def _fetch(self, url: str) -> dict:
        """Fetch GeoJSON from IEM LSR endpoint."""
        with wrap_errors(APIError, "Failed to fetch LSR data from IEM"):
            return tiny_retriever.fetch(url, "json", timeout=self.timeout)

    def _parse_features(self, geojson: dict) -> pd.DataFrame:
        """Parse GeoJSON features into a DataFrame, applying filters.

        Parameters
        ----------
        geojson : dict
            GeoJSON FeatureCollection from the IEM LSR endpoint.

        Returns
        -------
        pd.DataFrame
            Filtered storm report records.

        Raises
        ------
        DataNotFoundError
            If no features match the filters.
        """
        features = geojson.get("features", [])
        if not features:
            raise DataNotFoundError(
                f"LSR returned no features for {self.start_date} to {self.end_date}"
            )

        records = []
        for feature in features:
            props = feature.get("properties", {})
            geom = feature.get("geometry", {})

            # Filter by event type
            event_type = (props.get("typetext") or "").upper().strip()
            if not event_type:
                continue
            if self.event_types and event_type not in self.event_types:
                continue

            # Filter by state
            if self.state:
                event_state = (props.get("st") or "").upper().strip()
                if event_state != self.state:
                    continue

            # Extract coordinates from geometry
            coords = geom.get("coordinates", [None, None])

            records.append({
                "valid": props.get("valid"),
                "event_type": event_type,
                "magnitude": props.get("magnitude"),
                "unit": props.get("unit"),
                "wfo": props.get("wfo"),
                "county": props.get("county"),
                "state": props.get("st"),
                "city": props.get("city"),
                "source": props.get("source"),
                "remark": props.get("remark"),
                "longitude": coords[0] if coords else None,
                "latitude": coords[1] if len(coords) > 1 else None,
            })

        if not records:
            raise DataNotFoundError(
                f"No LSR features matched filters (event_types={self.event_types}, "
                f"state={self.state}) for {self.start_date} to {self.end_date}"
            )

        df = pd.DataFrame(records)
        df["valid"] = pd.to_datetime(df["valid"], errors="coerce")
        logger.debug(f"Parsed {len(df)} storm reports matching filters")
        return df

    def _reap(self) -> pd.DataFrame:
        """Fetch and parse NWS Local Storm Reports."""
        logger.info(
            f"Reaping LSR data: wfos={self.wfos or 'all'}, "
            f"event_types={self.event_types}, state={self.state or 'all'}"
        )

        url = self._build_url()
        geojson = self._fetch(url)
        df = self._parse_features(geojson)

        if self.transformations and not df.empty:
            df = apply_ts_transformations(df, self.transformations)

        logger.info(f"Reaped {len(df)} storm reports")
        return df
