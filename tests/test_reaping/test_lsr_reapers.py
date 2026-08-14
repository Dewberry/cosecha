"""Tests for LSR reaper."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from cosecha.exceptions import APIError, DataNotFoundError, DateRangeError
from cosecha.reaping.lsr import LSRReaper


SAMPLE_GEOJSON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [-105.0, 39.7]},
            "properties": {
                "valid": "2026-07-01T12:00:00Z",
                "typetext": "FLASH FLOOD",
                "magnitude": None,
                "unit": None,
                "wfo": "BOU",
                "county": "Denver",
                "st": "CO",
                "city": "Denver",
                "source": "Emergency Manager",
                "remark": "Water over road",
            },
        },
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [-104.5, 39.5]},
            "properties": {
                "valid": "2026-07-01T14:00:00Z",
                "typetext": "HEAVY RAIN",
                "magnitude": 2.5,
                "unit": "INCH",
                "wfo": "BOU",
                "county": "Arapahoe",
                "st": "CO",
                "city": "Aurora",
                "source": "CoCoRaHS",
                "remark": "Heavy rainfall",
            },
        },
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [-106.0, 40.0]},
            "properties": {
                "valid": "2026-07-01T15:00:00Z",
                "typetext": "HAIL",
                "magnitude": 1.0,
                "unit": "INCH",
                "wfo": "BOU",
                "county": "Boulder",
                "st": "CO",
                "city": "Boulder",
                "source": "Public",
                "remark": "Golf ball hail",
            },
        },
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [-102.0, 37.5]},
            "properties": {
                "valid": "2026-07-01T16:00:00Z",
                "typetext": "FLASH FLOOD",
                "magnitude": None,
                "unit": None,
                "wfo": "GLD",
                "county": "Baca",
                "st": "KS",
                "city": "Springfield",
                "source": "Law Enforcement",
                "remark": "Water over road",
            },
        },
    ],
}


class TestLSRReaper:
    """Tests for LSRReaper."""

    def test_initialization_valid(self):
        """Test valid initialization with all parameters."""
        reaper = LSRReaper(
            start_date="2026-07-01T00:00:00Z",
            end_date="2026-07-02T00:00:00Z",
            wfos=["BOU", "GJT"],
            event_types=["FLASH FLOOD", "HEAVY RAIN"],
            state="CO",
        )
        assert reaper.start_date == pd.Timestamp("2026-07-01T00:00:00", tz="UTC")
        assert reaper.end_date == pd.Timestamp("2026-07-02T00:00:00", tz="UTC")
        assert reaper.wfos == ["BOU", "GJT"]
        assert reaper.event_types == ["FLASH FLOOD", "HEAVY RAIN"]
        assert reaper.state == "CO"

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        reaper = LSRReaper(
            start_date="2026-07-01",
            end_date="2026-07-02",
        )
        assert reaper.wfos is None
        assert reaper.state is None
        assert reaper.event_types is None

    def test_invalid_date_range(self):
        """Test initialization fails with start > end."""
        with pytest.raises(DateRangeError):
            LSRReaper(
                start_date="2026-07-02",
                end_date="2026-07-01",
            )

    def test_invalid_date_format(self):
        """Test initialization fails with unparsable date."""
        with pytest.raises(DateRangeError, match="Could not parse date"):
            LSRReaper(
                start_date="not-a-date",
                end_date="2026-07-02",
            )

    def test_build_url_with_wfos(self):
        """Test URL construction includes WFOs."""
        reaper = LSRReaper(
            start_date="2026-07-01T00:00:00Z",
            end_date="2026-07-02T00:00:00Z",
            wfos=["BOU", "GJT"],
        )
        url = reaper._build_url()
        assert "sts=202607010000" in url
        assert "ets=202607020000" in url
        assert "wfos=BOU,GJT" in url

    def test_build_url_without_wfos(self):
        """Test URL construction without WFOs."""
        reaper = LSRReaper(
            start_date="2026-07-01T00:00:00Z",
            end_date="2026-07-02T00:00:00Z",
        )
        url = reaper._build_url()
        assert "wfos" not in url

    @patch("cosecha.reaping.lsr.tiny_retriever.fetch")
    def test_reap_success(self, mock_fetch):
        """Test successful data retrieval with filters."""
        mock_fetch.return_value = SAMPLE_GEOJSON

        reaper = LSRReaper(
            start_date="2026-07-01T00:00:00Z",
            end_date="2026-07-02T00:00:00Z",
            wfos=["BOU"],
            event_types=["FLASH FLOOD", "HEAVY RAIN"],
            state="CO",
        )
        df = reaper.reap()

        assert isinstance(df, pd.DataFrame)
        # HAIL and KS features should be filtered out
        assert len(df) == 2
        assert set(df["event_type"]) == {"FLASH FLOOD", "HEAVY RAIN"}
        assert all(df["state"] == "CO")

    @patch("cosecha.reaping.lsr.tiny_retriever.fetch")
    def test_reap_no_state_filter(self, mock_fetch):
        """Test retrieval without state filter returns more results."""
        mock_fetch.return_value = SAMPLE_GEOJSON

        reaper = LSRReaper(
            start_date="2026-07-01T00:00:00Z",
            end_date="2026-07-02T00:00:00Z",
            event_types=["FLASH FLOOD", "HEAVY RAIN"],
        )
        df = reaper.reap()

        assert len(df) == 3  # Both CO and KS flash floods + heavy rain

    @patch("cosecha.reaping.lsr.tiny_retriever.fetch")
    def test_reap_all_event_types(self, mock_fetch):
        """Test retrieval with no event_types filter returns all types."""
        mock_fetch.return_value = SAMPLE_GEOJSON

        reaper = LSRReaper(
            start_date="2026-07-01T00:00:00Z",
            end_date="2026-07-02T00:00:00Z",
        )
        df = reaper.reap()

        assert len(df) == 4  # All features including HAIL
        assert set(df["event_type"]) == {"FLASH FLOOD", "HEAVY RAIN", "HAIL"}

    @patch("cosecha.reaping.lsr.tiny_retriever.fetch")
    def test_reap_empty_features(self, mock_fetch):
        """Test raises DataNotFoundError when no features returned."""
        mock_fetch.return_value = {"type": "FeatureCollection", "features": []}

        reaper = LSRReaper(
            start_date="2026-07-01T00:00:00Z",
            end_date="2026-07-02T00:00:00Z",
        )
        with pytest.raises(DataNotFoundError, match="no features"):
            reaper.reap()

    @patch("cosecha.reaping.lsr.tiny_retriever.fetch")
    def test_reap_no_matching_features(self, mock_fetch):
        """Test raises DataNotFoundError when no features match filters."""
        mock_fetch.return_value = SAMPLE_GEOJSON

        reaper = LSRReaper(
            start_date="2026-07-01T00:00:00Z",
            end_date="2026-07-02T00:00:00Z",
            event_types=["TORNADO"],
        )
        with pytest.raises(DataNotFoundError, match="No LSR features matched"):
            reaper.reap()

    @patch("cosecha.reaping.lsr.tiny_retriever.fetch")
    def test_reap_api_error(self, mock_fetch):
        """Test wraps fetch errors as APIError."""
        mock_fetch.side_effect = Exception("Connection timeout")

        reaper = LSRReaper(
            start_date="2026-07-01T00:00:00Z",
            end_date="2026-07-02T00:00:00Z",
        )
        with pytest.raises(APIError, match="Failed to fetch LSR data"):
            reaper.reap()

    @patch("cosecha.reaping.lsr.tiny_retriever.fetch")
    def test_reap_with_transformations(self, mock_fetch):
        """Test transformations are applied to result."""
        mock_fetch.return_value = SAMPLE_GEOJSON

        reaper = LSRReaper(
            start_date="2026-07-01T00:00:00Z",
            end_date="2026-07-02T00:00:00Z",
            event_types=["FLASH FLOOD", "HEAVY RAIN"],
            state="CO",
            transformations={"rename_columns": {"event_type": "type"}},
        )
        df = reaper.reap()
        assert "type" in df.columns
        assert "event_type" not in df.columns

    @patch("cosecha.reaping.lsr.tiny_retriever.fetch")
    def test_coordinates_parsed(self, mock_fetch):
        """Test lat/lon are extracted from geometry."""
        mock_fetch.return_value = SAMPLE_GEOJSON

        reaper = LSRReaper(
            start_date="2026-07-01T00:00:00Z",
            end_date="2026-07-02T00:00:00Z",
            event_types=["FLASH FLOOD"],
            state="CO",
        )
        df = reaper.reap()
        assert df.iloc[0]["longitude"] == -105.0
        assert df.iloc[0]["latitude"] == 39.7
