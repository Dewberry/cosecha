"""Tests for USACE reservoir reapers."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from cosecha.exceptions import APIError, DateRangeError, InvalidSiteError
from cosecha.reaping.usace import ReservoirReaper


class TestReservoirReaper:
    """Tests for USACE ReservoirReaper."""

    def test_initialization_valid(self):
        """Test valid initialization."""
        reaper = ReservoirReaper(
            site_ids=["JPLT2"],
            params=["storage", "elevation"],
            start_date="2026-06-04T00:00:00Z",
            end_date="2026-06-05T00:00:00Z",
        )
        assert reaper.site_ids == ["JPLT2"]
        assert reaper.params == ["storage", "elevation"]
        assert reaper.start_date == pd.Timestamp("2026-06-04T00:00:00Z")
        assert reaper.end_date == pd.Timestamp("2026-06-05T00:00:00Z")
        assert reaper.provider == "swf"

    def test_initialization_custom_provider(self):
        """Test initialization with custom provider."""
        reaper = ReservoirReaper(
            site_ids=["JPLT2"],
            params=["storage"],
            start_date="2026-06-04T00:00:00Z",
            end_date="2026-06-05T00:00:00Z",
            provider="swl",
        )
        assert reaper.provider == "swl"

    def test_initialization_multiple_sites(self):
        """Test initialization with multiple sites."""
        reaper = ReservoirReaper(
            site_ids=["JPLT2", "BNBT2", "TBLT2"],
            params=["outflow"],
            start_date="2026-06-04T00:00:00Z",
            end_date="2026-06-05T00:00:00Z",
        )
        assert len(reaper.site_ids) == 3

    def test_empty_site_ids(self):
        """Test initialization fails with empty site IDs."""
        with pytest.raises(InvalidSiteError, match="site_ids cannot be empty"):
            ReservoirReaper(
                site_ids=[],
                params=["storage"],
                start_date="2026-06-04T00:00:00Z",
                end_date="2026-06-05T00:00:00Z",
            )

    def test_invalid_site_id(self):
        """Test initialization fails with invalid site ID."""
        with pytest.raises(InvalidSiteError, match="Invalid site ID"):
            ReservoirReaper(
                site_ids=[""],
                params=["storage"],
                start_date="2026-06-04T00:00:00Z",
                end_date="2026-06-05T00:00:00Z",
            )

    def test_empty_params(self):
        """Test initialization fails with empty params."""
        with pytest.raises(InvalidSiteError, match="params cannot be empty"):
            ReservoirReaper(
                site_ids=["JPLT2"],
                params=[],
                start_date="2026-06-04T00:00:00Z",
                end_date="2026-06-05T00:00:00Z",
            )

    def test_invalid_param(self):
        """Test initialization fails with unknown parameter."""
        with pytest.raises(InvalidSiteError, match="Unknown parameter"):
            ReservoirReaper(
                site_ids=["JPLT2"],
                params=["nonexistent"],
                start_date="2026-06-04T00:00:00Z",
                end_date="2026-06-05T00:00:00Z",
            )

    def test_invalid_date_range(self):
        """Test initialization fails with end before start."""
        with pytest.raises(DateRangeError):
            ReservoirReaper(
                site_ids=["JPLT2"],
                params=["storage"],
                start_date="2026-06-05T00:00:00Z",
                end_date="2026-06-04T00:00:00Z",
            )

    def test_invalid_start_date(self):
        """Test initialization fails with unparsable start date."""
        with pytest.raises(DateRangeError, match="Could not parse date"):
            ReservoirReaper(
                site_ids=["JPLT2"],
                params=["storage"],
                start_date="not-a-date",
                end_date="2026-06-05T00:00:00Z",
            )

    def test_invalid_end_date(self):
        """Test initialization fails with unparsable end date."""
        with pytest.raises(DateRangeError, match="Could not parse date"):
            ReservoirReaper(
                site_ids=["JPLT2"],
                params=["storage"],
                start_date="2026-06-04T00:00:00Z",
                end_date="nope",
            )

    # ------------------------------------------------------------------
    # Fetch / reap tests (mocked)
    # ------------------------------------------------------------------

    @patch("cosecha.reaping.usace.requests.get")
    def test_reap_success_single_site_single_param(self, mock_get):
        """Test successful fetch for one site and one parameter."""
        mock_response = mock_get.return_value
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "values": [
                ["2026-06-04T01:00:00Z", 150.0],
                ["2026-06-04T02:00:00Z", 152.0],
                ["2026-06-04T03:00:00Z", 151.5],
            ],
            "unit": "ac-ft",
        }

        reaper = ReservoirReaper(
            site_ids=["JPLT2"],
            params=["storage"],
            start_date="2026-06-04T00:00:00Z",
            end_date="2026-06-05T00:00:00Z",
        )
        harvested = reaper.reap()

        assert isinstance(harvested, pd.DataFrame)
        assert len(harvested) == 3
        assert list(harvested.columns) == ["datetime", "value", "site_id", "variable", "unit"]
        assert harvested["site_id"].unique().tolist() == ["JPLT2"]
        assert harvested["variable"].unique().tolist() == ["storage"]
        assert harvested["unit"].iloc[0] == "ac-ft"
        mock_get.assert_called_once()

    @patch("cosecha.reaping.usace.requests.get")
    def test_reap_multiple_sites_multiple_params(self, mock_get):
        """Test successful fetch for multiple sites and parameters."""
        mock_response = mock_get.return_value
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "values": [
                ["2026-06-04T01:00:00Z", 100.0],
                ["2026-06-04T02:00:00Z", 101.0],
            ],
            "unit": "ft",
        }

        reaper = ReservoirReaper(
            site_ids=["JPLT2", "BNBT2"],
            params=["storage", "elevation"],
            start_date="2026-06-04T00:00:00Z",
            end_date="2026-06-05T00:00:00Z",
        )
        harvested = reaper.reap()

        assert isinstance(harvested, pd.DataFrame)
        # 2 sites x 2 params x 2 records = 8
        assert len(harvested) == 8
        assert set(harvested["site_id"].unique()) == {"JPLT2", "BNBT2"}
        assert set(harvested["variable"].unique()) == {"storage", "elevation"}
        # Should be called 4 times (2 sites x 2 params)
        assert mock_get.call_count == 4

    @patch("cosecha.reaping.usace.requests.get")
    def test_reap_empty_response(self, mock_get):
        """Test reap handles empty API response."""
        mock_response = mock_get.return_value
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"values": []}

        reaper = ReservoirReaper(
            site_ids=["JPLT2"],
            params=["storage"],
            start_date="2026-06-04T00:00:00Z",
            end_date="2026-06-05T00:00:00Z",
        )
        harvested = reaper.reap()

        assert isinstance(harvested, pd.DataFrame)
        assert harvested.empty

    @patch("cosecha.reaping.usace.requests.get")
    def test_reap_no_values_key(self, mock_get):
        """Test reap handles response with missing values key."""
        mock_response = mock_get.return_value
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {}

        reaper = ReservoirReaper(
            site_ids=["JPLT2"],
            params=["storage"],
            start_date="2026-06-04T00:00:00Z",
            end_date="2026-06-05T00:00:00Z",
        )
        harvested = reaper.reap()

        assert isinstance(harvested, pd.DataFrame)
        assert harvested.empty

    @patch("cosecha.reaping.usace.requests.get")
    def test_reap_api_error(self, mock_get):
        """Test error handling when requests fails."""
        mock_get.side_effect = Exception("Connection timeout")

        reaper = ReservoirReaper(
            site_ids=["JPLT2"],
            params=["storage"],
            start_date="2026-06-04T00:00:00Z",
            end_date="2026-06-05T00:00:00Z",
        )

        with pytest.raises(APIError, match="Failed to fetch USACE time series"):
            reaper.reap()

    @patch("cosecha.reaping.usace.requests.get")
    def test_reap_http_error(self, mock_get):
        """Test error handling when API returns HTTP error."""
        from requests.exceptions import HTTPError

        mock_response = mock_get.return_value
        mock_response.raise_for_status.side_effect = HTTPError("404 Not Found")

        reaper = ReservoirReaper(
            site_ids=["JPLT2"],
            params=["storage"],
            start_date="2026-06-04T00:00:00Z",
            end_date="2026-06-05T00:00:00Z",
        )

        with pytest.raises(APIError, match="Failed to fetch USACE time series"):
            reaper.reap()

    @patch("cosecha.reaping.usace.requests.get")
    def test_reap_partial_failure(self, mock_get):
        """Test that partial failures still return available data."""
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            mock_resp = type("Response", (), {})()
            mock_resp.raise_for_status = lambda: None
            if call_count[0] == 1:
                # First call returns data
                mock_resp.json = lambda: {
                    "values": [["2026-06-04T01:00:00Z", 100.0]],
                    "unit": "ac-ft",
                }
            else:
                # Second call returns empty
                mock_resp.json = lambda: {"values": []}
            return mock_resp

        mock_get.side_effect = side_effect

        reaper = ReservoirReaper(
            site_ids=["JPLT2"],
            params=["storage", "elevation"],
            start_date="2026-06-04T00:00:00Z",
            end_date="2026-06-05T00:00:00Z",
        )
        harvested = reaper.reap()

        assert len(harvested) == 1
        assert harvested["variable"].iloc[0] == "storage"

    @patch("cosecha.reaping.usace.requests.get")
    def test_reap_stores_data_on_instance(self, mock_get):
        """Test that reap() stores data on the instance."""
        mock_response = mock_get.return_value
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "values": [["2026-06-04T01:00:00Z", 100.0]],
            "unit": "ac-ft",
        }

        reaper = ReservoirReaper(
            site_ids=["JPLT2"],
            params=["storage"],
            start_date="2026-06-04T00:00:00Z",
            end_date="2026-06-05T00:00:00Z",
        )
        assert reaper.data is None
        harvested = reaper.reap()
        assert reaper.data is harvested

    # ------------------------------------------------------------------
    # Transformation tests
    # ------------------------------------------------------------------

    @patch("cosecha.reaping.usace.requests.get")
    def test_reap_with_rename_columns(self, mock_get):
        """Test reap applies rename_columns transformation."""
        mock_response = mock_get.return_value
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "values": [
                ["2026-06-04T01:00:00Z", 150.0],
                ["2026-06-04T02:00:00Z", 152.0],
            ],
            "unit": "ac-ft",
        }

        reaper = ReservoirReaper(
            site_ids=["JPLT2"],
            params=["storage"],
            start_date="2026-06-04T00:00:00Z",
            end_date="2026-06-05T00:00:00Z",
            transformations={"rename_columns": {"value": "storage_value"}},
        )
        harvested = reaper.reap()

        assert "storage_value" in harvested.columns
        assert "value" not in harvested.columns

    @patch("cosecha.reaping.usace.requests.get")
    def test_reap_with_unit_conversions(self, mock_get):
        """Test reap applies unit_conversions transformation."""
        mock_response = mock_get.return_value
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "values": [
                ["2026-06-04T01:00:00Z", 100.0],
                ["2026-06-04T02:00:00Z", 200.0],
            ],
            "unit": "cfs",
        }

        reaper = ReservoirReaper(
            site_ids=["JPLT2"],
            params=["outflow"],
            start_date="2026-06-04T00:00:00Z",
            end_date="2026-06-05T00:00:00Z",
            transformations={"unit_conversions": {"value": 0.028316847}},  # cfs to cms
        )
        harvested = reaper.reap()

        assert harvested["value"].iloc[0] == pytest.approx(2.8316847)
        assert harvested["value"].iloc[1] == pytest.approx(5.6633694)

    @patch("cosecha.reaping.usace.requests.get")
    def test_reap_with_filter_columns(self, mock_get):
        """Test reap applies filter_columns transformation."""
        mock_response = mock_get.return_value
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "values": [
                ["2026-06-04T01:00:00Z", 150.0],
            ],
            "unit": "ac-ft",
        }

        reaper = ReservoirReaper(
            site_ids=["JPLT2"],
            params=["storage"],
            start_date="2026-06-04T00:00:00Z",
            end_date="2026-06-05T00:00:00Z",
            transformations={"filter_columns": ["site_id", "datetime", "value"]},
        )
        harvested = reaper.reap()

        assert list(harvested.columns) == ["site_id", "datetime", "value"]

    @patch("cosecha.reaping.usace.requests.get")
    def test_reap_with_multiple_transformations(self, mock_get):
        """Test reap applies multiple transformations in sequence."""
        mock_response = mock_get.return_value
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "values": [
                ["2026-06-04T01:00:00Z", 100.0],
            ],
            "unit": "cfs",
        }

        reaper = ReservoirReaper(
            site_ids=["JPLT2"],
            params=["outflow"],
            start_date="2026-06-04T00:00:00Z",
            end_date="2026-06-05T00:00:00Z",
            transformations={
                "unit_conversions": {"value": 2.0},
                "rename_columns": {"value": "flow"},
            },
        )
        harvested = reaper.reap()

        assert "flow" in harvested.columns
        assert harvested["flow"].iloc[0] == 200.0

    @patch("cosecha.reaping.usace.requests.get")
    def test_reap_no_transformations(self, mock_get):
        """Test reap with no transformations returns raw data."""
        mock_response = mock_get.return_value
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "values": [
                ["2026-06-04T01:00:00Z", 150.0],
            ],
            "unit": "ac-ft",
        }

        reaper = ReservoirReaper(
            site_ids=["JPLT2"],
            params=["storage"],
            start_date="2026-06-04T00:00:00Z",
            end_date="2026-06-05T00:00:00Z",
        )
        harvested = reaper.reap()

        assert list(harvested.columns) == ["datetime", "value", "site_id", "variable", "unit"]

    # ------------------------------------------------------------------
    # Request parameter tests
    # ------------------------------------------------------------------

    @patch("cosecha.reaping.usace.requests.get")
    def test_request_uses_correct_url(self, mock_get):
        """Test that the correct URL is constructed for the provider."""
        mock_response = mock_get.return_value
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"values": []}

        reaper = ReservoirReaper(
            site_ids=["JPLT2"],
            params=["storage"],
            start_date="2026-06-04T00:00:00Z",
            end_date="2026-06-05T00:00:00Z",
            provider="nwk",
        )
        reaper.reap()

        call_args = mock_get.call_args
        assert "nwk" in call_args[0][0] or "nwk" in call_args.kwargs.get("url", call_args[0][0])

    @patch("cosecha.reaping.usace.requests.get")
    def test_request_uses_correct_ts_name(self, mock_get):
        """Test that the correct time series name is sent."""
        mock_response = mock_get.return_value
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"values": []}

        reaper = ReservoirReaper(
            site_ids=["JPLT2"],
            params=["outflow"],
            start_date="2026-06-04T00:00:00Z",
            end_date="2026-06-05T00:00:00Z",
        )
        reaper.reap()

        call_args = mock_get.call_args
        params = call_args.kwargs.get("params", call_args[1].get("params"))
        assert params["name"] == "JPLT2-Gated_Total.Flow-Out.Inst.1Hour.0.Rev-SWF-REGI"

    @patch("cosecha.reaping.usace.requests.get")
    def test_request_timeout(self, mock_get):
        """Test that requests are made with a timeout."""
        mock_response = mock_get.return_value
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"values": []}

        reaper = ReservoirReaper(
            site_ids=["JPLT2"],
            params=["storage"],
            start_date="2026-06-04T00:00:00Z",
            end_date="2026-06-05T00:00:00Z",
        )
        reaper.reap()

        call_args = mock_get.call_args
        timeout = call_args.kwargs.get("timeout", call_args[1].get("timeout"))
        assert timeout == 30

    # ------------------------------------------------------------------
    # Network test (opt-in)
    # ------------------------------------------------------------------

    @pytest.mark.network
    def test_reap_network(self):
        """Test live network retrieval from USACE CDA."""
        reaper = ReservoirReaper(
            site_ids=["JPLT2"],
            params=["storage"],
            start_date="2026-06-04T00:00:00Z",
            end_date="2026-06-05T00:00:00Z",
        )
        harvested = reaper.reap()

        assert isinstance(harvested, pd.DataFrame)
        assert len(harvested) > 0
        assert "value" in harvested.columns
        assert "site_id" in harvested.columns
