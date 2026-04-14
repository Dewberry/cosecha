"""Tests for ASOS reapers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from cosecha.exceptions import APIError, DateRangeError, InvalidSiteError
from cosecha.reaping.asos import ASOSReaper


class TestASOSReaper:
    """Tests for ASOSReaper."""

    def test_initialization_valid(self):
        """Test valid initialization."""
        reaper = ASOSReaper(
            state="TX",
            data="p01i",
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
            data=["p01i", "tmpf"],
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
                data="p01i",
                start_date="2026-04-13",
                end_date="2026-04-12",
            )

    def test_invalid_start_date(self):
        """Test initialization fails with unparsable start date."""
        with pytest.raises(DateRangeError, match="Could not parse date"):
            ASOSReaper(
                state="TX",
                data="p01i",
                start_date="not-a-date",
                end_date="2026-04-13",
            )

    @pytest.mark.network
    def test_reap_network(self):
        """Test live network retrieval from ASOS."""
        reaper = ASOSReaper(
            state="TX",
            data="p01i",
            start_date="2022-01-01",
            end_date="2022-01-02",
        )
        harvested = reaper.reap()
        assert len(harvested) > 0
        assert "p01i" in harvested.columns
        assert isinstance(harvested, pd.DataFrame)

    @patch("cosecha.reaping.asos.requests.get")
    def test_reap_success(self, mock_get):
        """Test successful data retrieval."""
        mock_response = MagicMock()
        mock_response.text = (
            "# comment 1\n"
            "# comment 2\n"
            "# comment 3\n"
            "# comment 4\n"
            "# comment 5\n"
            "station,valid,lon,lat,p01i\n"
            "AUS,2026-04-12 00:00,-97.6698,30.1945,0.01\n"
            "AUS,2026-04-12 01:00,-97.6698,30.1945,0.02\n"
        )
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        reaper = ASOSReaper(
            state="TX",
            data="p01i",
            start_date="2026-04-12",
            end_date="2026-04-12",
        )

        harvested = reaper.reap()

        assert len(harvested) == 2
        assert isinstance(harvested, pd.DataFrame)
        assert "p01i" in harvested.columns
        assert "station" in harvested.columns
        mock_get.assert_called_once()
        
        args, kwargs = mock_get.call_args
        assert kwargs["params"]["network"] == "TX_ASOS"
        assert kwargs["params"]["data"] == ["p01i"]

    @patch("cosecha.reaping.asos.time.sleep")
    @patch("cosecha.reaping.asos.requests.get")
    def test_reap_retry_logic(self, mock_get, mock_sleep):
        """Test retry logic when API fails temporarily."""
        mock_response = MagicMock()
        mock_response.text = (
            "# 1\n# 2\n# 3\n# 4\n# 5\n"
            "station,p01i\n"
            "AUS,0.01\n"
        )
        
        # Fail twice, succeed on third
        mock_get.side_effect = [
            requests.RequestException("Timeout 1"),
            requests.RequestException("Timeout 2"),
            mock_response
        ]

        reaper = ASOSReaper(
            state="TX",
            start_date="2026-04-12",
            end_date="2026-04-12",
        )

        harvested = reaper.reap()
        
        assert len(harvested) == 1
        assert mock_get.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("cosecha.reaping.asos.time.sleep")
    @patch("cosecha.reaping.asos.requests.get")
    def test_reap_api_error(self, mock_get, mock_sleep):
        """Test error handling when API fails permanently."""
        mock_get.side_effect = requests.RequestException("Connection failed")

        reaper = ASOSReaper(
            state="TX",
            start_date="2026-04-12",
            end_date="2026-04-12",
        )

        with pytest.raises(APIError, match="Failed to fetch ASOS data for TX_ASOS"):
            reaper.reap()
            
        assert mock_get.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("cosecha.reaping.asos.requests.get")
    def test_reap_empty_response(self, mock_get):
        """Test reap handles empty output after skipping comments."""
        mock_response = MagicMock()
        # Ensure that parsing returns an empty dataframe (e.g., just comments or header only)
        mock_response.text = (
            "# 1\n# 2\n# 3\n# 4\n# 5\n"
            "station,valid,lon,lat,p01i\n"
        )
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        reaper = ASOSReaper(
            state="TX",
            start_date="2026-04-12",
            end_date="2026-04-12",
        )

        harvested = reaper.reap()
        assert isinstance(harvested, pd.DataFrame)
        assert harvested.empty

    @patch("cosecha.reaping.asos.requests.get")
    def test_reap_with_transformations(self, mock_get):
        """Test reap applies format transformations when not empty."""
        mock_response = MagicMock()
        mock_response.text = (
            "# 1\n# 2\n# 3\n# 4\n# 5\n"
            "station,p01i\n"
            "AUS,0.01\n"
        )
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        reaper = ASOSReaper(
            state="TX",
            start_date="2026-04-12",
            end_date="2026-04-12",
            transformations={"rename_columns": {"p01i": "precip"}},
        )

        harvested = reaper.reap()
        assert "precip" in harvested.columns
        assert "p01i" not in harvested.columns
