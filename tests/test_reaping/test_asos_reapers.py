"""Tests for ASOS reapers."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from cosecha.exceptions import APIError, DateRangeError, InvalidSiteError
from cosecha.reaping.asos import ASOSReaper


class TestASOSReaper:
    """Tests for ASOSReaper."""

    def test_initialization_valid(self):
        """Test valid initialization."""
        reaper = ASOSReaper(
            state="TX",
            variable="p01i",
            start_date="2026-04-12",
            end_date="2026-04-13",
        )
        assert reaper.state == "TX"
        assert reaper.network == "TX_ASOS"
        assert reaper.start_date == pd.Timestamp("2026-04-12")
        assert reaper.end_date == pd.Timestamp("2026-04-13")
        assert reaper.data_vars == ["p01i"]

    def test_initialization_multiple_data_vars(self):
        """Test initialization with multiple data variables."""
        reaper = ASOSReaper(
            state="IA",
            variable=["p01i", "tmpf"],
            start_date="2026-04-12",
            end_date="2026-04-13",
        )
        assert reaper.data_vars == ["p01i", "tmpf"]
        assert reaper.network == "IA_ASOS"

    def test_initialization_all_network(self):
        """Test initialization with ASOS state."""
        reaper = ASOSReaper(
            state="ASOS",
            start_date="2026-04-12",
            end_date="2026-04-13",
        )
        assert reaper.network == "ASOS"
        assert reaper.data_vars == ["all"]

    def test_empty_state(self):
        """Test initialization fails with empty state."""
        with pytest.raises(InvalidSiteError, match="state cannot be empty"):
            ASOSReaper(
                state="",
                start_date="2026-04-12",
                end_date="2026-04-13",
            )

    def test_invalid_date_range(self):
        """Test initialization fails with invalid date range."""
        with pytest.raises(DateRangeError):
            ASOSReaper(
                state="TX",
                variable="p01i",
                start_date="2026-04-13",
                end_date="2026-04-12",
            )

    def test_invalid_start_date(self):
        """Test initialization fails with unparsable start date."""
        with pytest.raises(DateRangeError, match="Could not parse date"):
            ASOSReaper(
                state="TX",
                variable="p01i",
                start_date="not-a-date",
                end_date="2026-04-13",
            )

    @pytest.mark.network
    def test_reap_network(self):
        """Test live network retrieval from ASOS."""
        reaper = ASOSReaper(
            state="TX",
            variable="p01i",
            start_date="2022-01-01",
            end_date="2022-01-02",
        )
        harvested = reaper.reap()
        assert len(harvested) > 0
        assert "p01i" in harvested.columns
        assert isinstance(harvested, pd.DataFrame)

    @patch("cosecha.reaping.asos.tiny_retriever.fetch")
    def test_reap_success(self, mock_fetch):
        """Test successful data retrieval."""
        mock_fetch.return_value = (
            "# comment 1\n"
            "# comment 2\n"
            "# comment 3\n"
            "# comment 4\n"
            "# comment 5\n"
            "station,valid,lon,lat,p01i\n"
            "AUS,2026-04-12 00:00,-97.6698,30.1945,0.01\n"
            "AUS,2026-04-12 01:00,-97.6698,30.1945,0.02\n"
        )

        reaper = ASOSReaper(
            state="TX",
            variable="p01i",
            start_date="2026-04-12",
            end_date="2026-04-12",
        )

        harvested = reaper.reap()

        assert len(harvested) == 2
        assert isinstance(harvested, pd.DataFrame)
        assert "p01i" in harvested.columns
        assert "station" in harvested.columns
        mock_fetch.assert_called_once()

        url = mock_fetch.call_args[0][0]
        assert "network=TX_ASOS" in url
        assert "data=p01i" in url

    @patch("cosecha.reaping.asos.tiny_retriever.fetch")
    def test_reap_api_error(self, mock_fetch):
        """Test error handling when API fails."""
        mock_fetch.side_effect = Exception("Connection failed")

        reaper = ASOSReaper(
            state="TX",
            start_date="2026-04-12",
            end_date="2026-04-12",
        )

        with pytest.raises(APIError, match="Failed to fetch ASOS data for TX_ASOS"):
            reaper.reap()

        mock_fetch.assert_called_once()

    @patch("cosecha.reaping.asos.tiny_retriever.fetch")
    def test_reap_empty_response(self, mock_fetch):
        """Test reap handles empty output after skipping comments."""
        mock_fetch.return_value = (
            "# 1\n# 2\n# 3\n# 4\n# 5\n"
            "station,valid,lon,lat,p01i\n"
        )

        reaper = ASOSReaper(
            state="TX",
            start_date="2026-04-12",
            end_date="2026-04-12",
        )

        harvested = reaper.reap()
        assert isinstance(harvested, pd.DataFrame)
        assert harvested.empty

    @patch("cosecha.reaping.asos.tiny_retriever.fetch")
    def test_reap_with_transformations(self, mock_fetch):
        """Test reap applies format transformations when not empty."""
        mock_fetch.return_value = (
            "# 1\n# 2\n# 3\n# 4\n# 5\n"
            "station,p01i\n"
            "AUS,0.01\n"
        )

        reaper = ASOSReaper(
            state="TX",
            start_date="2026-04-12",
            end_date="2026-04-12",
            transformations={"rename_columns": {"p01i": "precip"}},
        )

        harvested = reaper.reap()
        assert "precip" in harvested.columns
        assert "p01i" not in harvested.columns
