"""Tests for USGS NWIS reapers."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from cosecha.reaping.exceptions import APIError, DateRangeError, InvalidSiteError
from cosecha.reaping.nwis import USGSPrecipReaper, USGSStageReaper, USGSStreamflowReaper


class TestUSGSStreamflowReaper:
    """Tests for USGSStreamflowReaper."""

    def test_initialization_valid(self):
        """Test valid initialization."""
        reaper = USGSStreamflowReaper(
            site_ids=["01018035"],
            start_date="2026-01-01",
            end_date="2026-01-31",
        )
        assert reaper.site_ids == ["01018035"]
        assert reaper.start_date == "2026-01-01"
        assert reaper.end_date == "2026-01-31"
        assert reaper.parameter_code == "00060"
        assert reaper.stat_code == "00003"

    def test_initialization_multiple_sites(self):
        """Test initialization with multiple sites."""
        reaper = USGSStreamflowReaper(
            site_ids=["01018035", "01040000"],
            start_date="2026-01-01",
            end_date="2026-01-31",
        )
        assert len(reaper.site_ids) == 2

    def test_empty_site_ids(self):
        """Test initialization fails with empty site IDs."""
        with pytest.raises(InvalidSiteError):
            USGSStreamflowReaper(
                site_ids=[],
                start_date="2026-01-01",
                end_date="2026-01-31",
            )

    def test_invalid_site_id(self):
        """Test initialization fails with invalid site ID."""
        with pytest.raises(InvalidSiteError):
            USGSStreamflowReaper(
                site_ids=[""],
                start_date="2026-01-01",
                end_date="2026-01-31",
            )

    def test_invalid_date_range(self):
        """Test initialization fails with invalid date range."""
        with pytest.raises(DateRangeError):
            USGSStreamflowReaper(
                site_ids=["01018035"],
                start_date="2026-01-31",
                end_date="2026-01-01",
            )

    def test_invalid_start_date(self):
        """Test initialization fails with invalid start date format."""
        with pytest.raises(DateRangeError):
            USGSStreamflowReaper(
                site_ids=["01018035"],
                start_date="2026/01/01",
                end_date="2026-01-31",
            )

    def test_get_variable_name(self):
        """Test variable name for streamflow."""
        reaper = USGSStreamflowReaper(
            site_ids=["01018035"],
            start_date="2026-01-01",
            end_date="2026-01-31",
        )
        assert reaper._get_variable_name() == "streamflow"

    def test_get_data_type(self):
        """Test data type for streamflow."""
        reaper = USGSStreamflowReaper(
            site_ids=["01018035"],
            start_date="2026-01-01",
            end_date="2026-01-31",
        )
        assert reaper._get_data_type() == "instantaneous streamflow"

    @patch("cosecha.reaping.nwis.dr_nwis.get_iv")
    def test_reap_success(self, mock_get_iv):
        """Test successful data retrieval using dataretrieval."""
        # Mock dataretrieval response
        mock_df = pd.DataFrame(
            {
                "site_no": ["01018035", "01018035"],
                "00060": [198.0, 195.0],
                "00060_cd": ["A, e", "A, e"],
            }
        )
        mock_df.index = pd.to_datetime(["2022-01-01 15:00:00+00:00", "2022-01-02 03:00:00+00:00"])
        mock_df.index.name = "datetime"
        mock_metadata = {}

        mock_get_iv.return_value = (mock_df, mock_metadata)

        reaper = USGSStreamflowReaper(
            site_ids=["01018035"],
            start_date="2022-01-01",
            end_date="2022-01-31",
        )

        harvested = reaper.reap()

        assert len(harvested.data) == 2
        assert harvested.source_name == "USGS_NWIS"
        assert harvested.variable_names == ["streamflow"]
        mock_get_iv.assert_called_once()

    @patch("cosecha.reaping.nwis.dr_nwis.get_iv")
    def test_reap_multiple_sites(self, mock_get_iv):
        """Test retrieval for multiple sites."""
        mock_df = pd.DataFrame(
            {
                "site_no": ["01018035", "01040000"],
                "00060": [198.0, 150.0],
                "00060_cd": ["A, e", "A, e"],
            }
        )
        mock_df.index = pd.to_datetime(["2022-01-01 15:00:00+00:00", "2022-01-01 15:00:00+00:00"])
        mock_df.index.name = "datetime"

        mock_get_iv.return_value = (mock_df, {})

        reaper = USGSStreamflowReaper(
            site_ids=["01018035", "01040000"],
            start_date="2022-01-01",
            end_date="2022-01-31",
        )

        harvested = reaper.reap()

        assert len(harvested.data) == 2
        assert harvested.metadata["sites"] == ["01018035", "01040000"]

    @patch("cosecha.reaping.nwis.dr_nwis.get_iv")
    def test_reap_api_error(self, mock_get_iv):
        """Test error handling when dataretrieval fails."""
        mock_get_iv.side_effect = Exception("API connection failed")

        reaper = USGSStreamflowReaper(
            site_ids=["01018035"],
            start_date="2022-01-01",
            end_date="2022-01-31",
        )

        with pytest.raises(APIError, match="Failed to fetch NWIS data"):
            reaper.reap()


class TestUSGSStageReaper:
    """Tests for USGSStageReaper."""

    def test_initialization(self):
        """Test valid initialization."""
        reaper = USGSStageReaper(
            site_ids=["01018035"],
            start_date="2026-01-01",
            end_date="2026-01-31",
        )
        assert reaper.parameter_code == "00065"

    def test_get_variable_name(self):
        """Test variable name for stage."""
        reaper = USGSStageReaper(
            site_ids=["01018035"],
            start_date="2026-01-01",
            end_date="2026-01-31",
        )
        assert reaper._get_variable_name() == "stage"


class TestUSGSPrecipReaper:
    """Tests for USGSPrecipReaper."""

    def test_initialization(self):
        """Test valid initialization."""
        reaper = USGSPrecipReaper(
            site_ids=["01018035"],
            start_date="2026-01-01",
            end_date="2026-01-31",
        )
        assert reaper.parameter_code == "00045"
        assert reaper.stat_code == "00006"  # Sum for precip

    def test_get_variable_name(self):
        """Test variable name for precipitation."""
        reaper = USGSPrecipReaper(
            site_ids=["01018035"],
            start_date="2026-01-01",
            end_date="2026-01-31",
        )
        assert reaper._get_variable_name() == "precipitation"
