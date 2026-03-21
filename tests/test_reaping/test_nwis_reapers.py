"""Tests for USGS NWIS reapers."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from cosecha.reaping.exceptions import APIError, DateRangeError, InvalidSiteError
from cosecha.reaping.nwis import (
    USGSPrecipReaper,
    USGSStageReaper,
    USGSStreamflowReaper,
)


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

    def test_build_url(self):
        """Test URL building."""
        reaper = USGSStreamflowReaper(
            site_ids=["01018035"],
            start_date="2026-01-01",
            end_date="2026-01-31",
        )
        url = reaper._build_url()
        assert "sites=01018035" in url
        assert "startDT=2026-01-01" in url
        assert "endDT=2026-01-31" in url
        assert "parameterCd=00060" in url
        assert "statCd=00003" in url
        assert "format=json" in url

    def test_build_url_multiple_sites(self):
        """Test URL building with multiple sites."""
        reaper = USGSStreamflowReaper(
            site_ids=["01018035", "01040000"],
            start_date="2026-01-01",
            end_date="2026-01-31",
        )
        url = reaper._build_url()
        assert "sites=01018035,01040000" in url

    def test_parse_response_valid(self):
        """Test parsing a valid NWIS response."""
        reaper = USGSStreamflowReaper(
            site_ids=["01018035"],
            start_date="2026-01-01",
            end_date="2026-01-31",
        )
        response = {
            "value": {
                "timeSeries": [
                    {
                        "sourceInfo": {
                            "siteCd": [{"value": "01018035"}],
                        },
                        "values": [
                            {
                                "value": [
                                    {"dateTime": "2026-01-01T00:00:00.000", "value": "1.5"},
                                    {"dateTime": "2026-01-02T00:00:00.000", "value": "1.6"},
                                ]
                            }
                        ],
                    }
                ]
            }
        }
        df = reaper._parse_response(response)
        assert len(df) == 2
        assert list(df.columns) == ["time", "site_id", "value"]
        assert df["site_id"].iloc[0] == "01018035"
        assert df["value"].iloc[0] == 1.5

    def test_parse_response_multiple_sites(self):
        """Test parsing response with multiple sites."""
        reaper = USGSStreamflowReaper(
            site_ids=["01018035", "01040000"],
            start_date="2026-01-01",
            end_date="2026-01-31",
        )
        response = {
            "value": {
                "timeSeries": [
                    {
                        "sourceInfo": {
                            "siteCd": [{"value": "01018035"}],
                        },
                        "values": [
                            {
                                "value": [
                                    {"dateTime": "2026-01-01T00:00:00.000", "value": "1.5"},
                                ]
                            }
                        ],
                    },
                    {
                        "sourceInfo": {
                            "siteCd": [{"value": "01040000"}],
                        },
                        "values": [
                            {
                                "value": [
                                    {"dateTime": "2026-01-01T00:00:00.000", "value": "2.0"},
                                ]
                            }
                        ],
                    },
                ]
            }
        }
        df = reaper._parse_response(response)
        assert len(df) == 2
        assert df[df["site_id"] == "01018035"].iloc[0]["value"] == 1.5
        assert df[df["site_id"] == "01040000"].iloc[0]["value"] == 2.0

    def test_parse_response_with_null_values(self):
        """Test parsing response with null values."""
        reaper = USGSStreamflowReaper(
            site_ids=["01018035"],
            start_date="2026-01-01",
            end_date="2026-01-31",
        )
        response = {
            "value": {
                "timeSeries": [
                    {
                        "sourceInfo": {
                            "siteCd": [{"value": "01018035"}],
                        },
                        "values": [
                            {
                                "value": [
                                    {"dateTime": "2026-01-01T00:00:00.000", "value": "1.5"},
                                    {"dateTime": "2026-01-02T00:00:00.000", "value": None},
                                ]
                            }
                        ],
                    }
                ]
            }
        }
        df = reaper._parse_response(response)
        assert len(df) == 2
        assert pd.notna(df["value"].iloc[0])
        assert pd.isna(df["value"].iloc[1])

    def test_parse_response_empty_data(self):
        """Test parsing response with no data values."""
        reaper = USGSStreamflowReaper(
            site_ids=["01018035"],
            start_date="2026-01-01",
            end_date="2026-01-31",
        )
        response = {
            "value": {
                "timeSeries": [
                    {
                        "sourceInfo": {
                            "siteCd": [{"value": "01018035"}],
                        },
                        "values": [
                            {
                                "value": []
                            }
                        ],
                    }
                ]
            }
        }
        df = reaper._parse_response(response)
        assert len(df) == 0

    def test_parse_response_missing_value_key(self):
        """Test parsing fails with missing 'value' key."""
        reaper = USGSStreamflowReaper(
            site_ids=["01018035"],
            start_date="2026-01-01",
            end_date="2026-01-31",
        )
        response = {}
        with pytest.raises(APIError, match="missing 'value' key"):
            reaper._parse_response(response)

    def test_parse_response_no_timeseries(self):
        """Test parsing fails with no timeSeries."""
        reaper = USGSStreamflowReaper(
            site_ids=["01018035"],
            start_date="2026-01-01",
            end_date="2026-01-31",
        )
        response = {"value": {"timeSeries": []}}
        with pytest.raises(APIError, match="No time series data"):
            reaper._parse_response(response)

    def test_get_variable_name(self):
        """Test variable name for streamflow."""
        reaper = USGSStreamflowReaper(
            site_ids=["01018035"],
            start_date="2026-01-01",
            end_date="2026-01-31",
        )
        assert reaper._get_variable_name() == "streamflow"


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


class TestReaperReapMethod:
    """Tests for the reap() method (with mocking)."""

    def test_reap_creates_harvested_data(self, mocker):
        """Test that reap() returns HarvestedData with proper structure."""
        # Mock the HTTP request
        mock_response = {
            "value": {
                "timeSeries": [
                    {
                        "sourceInfo": {
                            "siteCd": [{"value": "01018035"}],
                        },
                        "values": [
                            {
                                "value": [
                                    {"dateTime": "2026-01-01T00:00:00.000", "value": "1.5"},
                                ]
                            }
                        ],
                    }
                ]
            }
        }
        mocker.patch.object(
            USGSStreamflowReaper,
            '_fetch_with_retry',
            return_value=mock_response,
        )

        reaper = USGSStreamflowReaper(
            site_ids=["01018035"],
            start_date="2026-01-01",
            end_date="2026-01-31",
        )
        data = reaper.reap()

        assert data.source_name == "USGS_NWIS"
        assert data.is_timeseries()
        assert "streamflow" in data.variable_names
        assert data.metadata["record_count"] == 1
        assert len(data.data) == 1

    def test_reap_with_api_failure(self, mocker):
        """Test that reap() raises APIError on fetch failure."""
        mocker.patch.object(
            USGSStreamflowReaper,
            '_fetch_with_retry',
            side_effect=APIError("API error"),
        )

        reaper = USGSStreamflowReaper(
            site_ids=["01018035"],
            start_date="2026-01-01",
            end_date="2026-01-31",
        )
        with pytest.raises(APIError):
            reaper.reap()
