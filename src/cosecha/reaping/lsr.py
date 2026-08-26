"""NWS Local Storm Reports (LSR) reaper.

This module provides an implementation for harvesting NWS Local Storm Reports
from the Iowa Environmental Mesonet (IEM) GeoJSON API at
https://mesonet.agron.iastate.edu/geojson/lsr.php
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlencode

import pandas as pd
import tiny_retriever

from cosecha._logging import logger
from cosecha._utils import apply_ts_transformations, parse_date_range, wrap_errors
from cosecha.exceptions import APIError, DataNotFoundError
from cosecha.reaping.base import TimeSeriesReaper

__all__ = [
    "LSRReaper",
]

BASE_URL = "https://mesonet.agron.iastate.edu/geojson/lsr.php"
_IEM_ROW_CAP = 10_000

_COLUMNS = {
    "valid": "valid",
    "typetext": "event_type",
    "magf": "magnitude",
    "unit": "unit",
    "wfo": "wfo",
    "county": "county",
    "st": "state",
    "city": "city",
    "source": "source",
    "remark": "remark",
    "product_id": "product_id",
    "lon": "longitude",
    "lat": "latitude",
}


class LSRReaper(TimeSeriesReaper):
    """Reaper for NWS Local Storm Reports via the IEM GeoJSON endpoint.

    Fetches storm report features for a given time window, optionally filtering
    by WFO (Weather Forecast Office), event type, and state.
    """

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
            If None, no state filtering is applied.  When provided the filter
            is applied both server-side (geographic intersection via IEM's
            ``states`` parameter) and client-side (exact ``st`` match), so
            border reports that intersect the state polygon but carry a
            neighbouring state code are excluded.
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

        self.start_date, self.end_date = parse_date_range(start_date, end_date)

        self.wfos = [w.upper().strip() for w in wfos] if wfos else None
        self.event_types = [et.upper().strip() for et in event_types] if event_types else None
        self.state = state.upper().strip() if state else None
        self.transformations = transformations
        self.timeout = timeout

        logger.debug(
            f"Initialized {self.__class__.__name__}: "
            f"wfos={self.wfos}, event_types={self.event_types}, state={self.state}, "
            f"dates={self.start_date} to {self.end_date}"
        )

    def _validate_params(self) -> None:
        """Date validation is handled by parse_date_range at construction."""

    def _build_url(self, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> str:
        """Build the IEM LSR GeoJSON request URL.

        Parameters
        ----------
        start, end : pd.Timestamp, optional
            Override the instance dates (used by ``_fetch_windows`` when
            bisecting capped responses).
        """
        s = start if start is not None else self.start_date
        e = end if end is not None else self.end_date
        params: dict[str, str] = {
            "sts": s.strftime("%Y%m%d%H%M"),
            "ets": e.strftime("%Y%m%d%H%M"),
        }
        if self.wfos:
            params["wfos"] = ",".join(self.wfos)
        if self.state:
            params["states"] = self.state
        return f"{BASE_URL}?{urlencode(params)}"

    def _fetch(self, urls: list[str]) -> list[dict[str, Any]]:
        """Fetch GeoJSON from IEM LSR endpoint.

        ``tiny_retriever`` runs the URLs concurrently and returns results in
        order.
        """
        with wrap_errors(APIError, "Failed to fetch LSR data from IEM"):
            return tiny_retriever.fetch(urls, "json", timeout=self.timeout)

    def _fetch_windows(
        self,
        windows: list[tuple[pd.Timestamp, pd.Timestamp]],
    ) -> list[dict[str, Any]]:
        """Fetch windows concurrently, bisecting any that hit the IEM row cap."""
        features: list[dict[str, Any]] = []
        pending = list(windows)
        while pending:
            responses = self._fetch([self._build_url(s, e) for s, e in pending])
            next_round: list[tuple[pd.Timestamp, pd.Timestamp]] = []
            # strict: a length mismatch would silently misalign windows with responses.
            for (start, end), response in zip(pending, responses, strict=True):
                chunk = response.get("features", [])
                if len(chunk) < _IEM_ROW_CAP:
                    features.extend(chunk)
                    continue
                # IEM matches ``valid BETWEEN sts AND ets``, inclusive at both ends, and
                # reports are stored at minute resolution (as is the sts/ets format). So
                # the second chunk starts one minute after the first ends: the halves are
                # exactly disjoint, so nothing is missed and nothing needs deduplicating.
                mid = (start + (end - start) / 2).floor("min")
                if mid <= start or mid >= end:
                    raise APIError(
                        f"LSR window {start} to {end} hits IEM's {_IEM_ROW_CAP}-row cap "
                        "and cannot be split further. Narrow the query with 'state' or 'wfos'."
                    )
                logger.debug(f"Window {start}-{end} hit row cap, bisecting at {mid}")
                next_round += [(start, mid), (mid + pd.Timedelta(minutes=1), end)]
            pending = next_round
        return features

    def _parse_features(self, features: list[dict[str, Any]]) -> pd.DataFrame:
        """Parse GeoJSON features into a DataFrame, applying filters.

        Parameters
        ----------
        features : list[dict]
            GeoJSON Feature dicts from the IEM LSR endpoint.

        Returns
        -------
        pd.DataFrame
            Filtered storm report records.

        Raises
        ------
        DataNotFoundError
            If no features match the filters.
        """
        if not features:
            raise DataNotFoundError(
                f"LSR returned no features for {self.start_date} to {self.end_date}"
            )

        df = pd.json_normalize(features)
        props_cols = [f"properties.{c}" for c in _COLUMNS]
        missing = [c for c in props_cols if c not in df.columns]
        if missing:
            raise APIError(
                f"IEM response missing expected properties: {[c.split('.', 1)[1] for c in missing]}"
            )
        df = df[props_cols]
        df.columns = list(_COLUMNS.values())

        df["valid"] = pd.to_datetime(df["valid"], utc=True, errors="coerce")
        df["event_type"] = df["event_type"].str.upper().str.strip()

        if self.event_types:
            df = df[df["event_type"].isin(self.event_types)]
        if self.state:
            df = df[df["state"].str.upper().str.strip() == self.state]

        if df.empty:
            raise DataNotFoundError(
                f"No LSR features matched filters (event_types={self.event_types}, "
                f"state={self.state}) for {self.start_date} to {self.end_date}"
            )

        logger.debug(f"Parsed {len(df)} storm reports matching filters")
        return df.reset_index(drop=True)

    def _reap(self) -> pd.DataFrame:
        """Fetch and parse NWS Local Storm Reports."""
        logger.info(
            f"Reaping LSR data: wfos={self.wfos or 'all'}, "
            f"event_types={self.event_types}, state={self.state or 'all'}"
        )

        features = self._fetch_windows([(self.start_date, self.end_date)])
        df = self._parse_features(features)

        if self.transformations and not df.empty:
            df = apply_ts_transformations(df, self.transformations)

        logger.info(f"Reaped {len(df)} storm reports")
        return df
