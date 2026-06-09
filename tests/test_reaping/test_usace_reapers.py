"""Tests for USACE reservoir reapers."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest
from tiny_retriever.exceptions import ServiceError

from cosecha.exceptions import APIError, DateRangeError, InvalidSiteError
from cosecha.reaping.usace import ReservoirReaper


def _make_catalog(site_ids, params):
    """Build a fake locations catalog for the given sites and params."""
    label_map = {
        "storage": ("Flood Storage", "Stor.Inst.1Hour.0.Decodes-Rev", "ac-ft"),
        "elevation": ("Elevation", "Elev.Inst.1Hour.0.Decodes-Rev", "ft"),
        "outflow": ("Outflow", "Flow-Out.Inst.1Hour.0.Rev-SWF-REGI", "cfs"),
        "inflow": ("Inflow", "Flow-In.Ave.~1Day.1Day.Computed-SWF-REGI", "cfs"),
    }
    catalog = []
    for site_id in site_ids:
        ts_list = []
        for param in params:
            label, suffix, unit = label_map[param]
            ts_list.append(
                {
                    "tsid": f"{site_id}.{suffix}",
                    "label": label,
                    "parameter": param,
                    "unit": unit,
                }
            )
        catalog.append({"code": site_id, "timeseries": ts_list})
    return catalog


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

    @patch("cosecha.reaping.usace.tiny_retriever.fetch")
    def test_reap_success_single_site_single_param(self, mock_fetch):
        """Test successful fetch for one site and one parameter."""
        catalog = _make_catalog(["JPLT2"], ["storage"])
        ts_response = [
            {
                "values": [
                    ["2026-06-04T01:00:00Z", 150.0],
                    ["2026-06-04T02:00:00Z", 152.0],
                    ["2026-06-04T03:00:00Z", 151.5],
                ],
                "unit": "ac-ft",
            }
        ]
        mock_fetch.side_effect = [[catalog], ts_response]

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

    @patch("cosecha.reaping.usace.tiny_retriever.fetch")
    def test_reap_multiple_sites_multiple_params(self, mock_fetch):
        """Test successful fetch for multiple sites and parameters."""
        catalog = _make_catalog(["JPLT2", "BNBT2"], ["storage", "elevation"])
        ts_response = [
            {
                "values": [
                    ["2026-06-04T01:00:00Z", 100.0],
                    ["2026-06-04T02:00:00Z", 101.0],
                ],
                "unit": "ft",
            }
        ] * 4  # 2 sites x 2 params
        mock_fetch.side_effect = [[catalog], ts_response]

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

    @patch("cosecha.reaping.usace.tiny_retriever.fetch")
    def test_reap_empty_response(self, mock_fetch):
        """Test reap handles empty API response."""
        catalog = _make_catalog(["JPLT2"], ["storage"])
        mock_fetch.side_effect = [[catalog], [{"values": []}]]

        reaper = ReservoirReaper(
            site_ids=["JPLT2"],
            params=["storage"],
            start_date="2026-06-04T00:00:00Z",
            end_date="2026-06-05T00:00:00Z",
        )
        harvested = reaper.reap()

        assert isinstance(harvested, pd.DataFrame)
        assert harvested.empty

    @patch("cosecha.reaping.usace.tiny_retriever.fetch")
    def test_reap_no_values_key(self, mock_fetch):
        """Test reap handles response with missing values key."""
        catalog = _make_catalog(["JPLT2"], ["storage"])
        mock_fetch.side_effect = [[catalog], [{}]]

        reaper = ReservoirReaper(
            site_ids=["JPLT2"],
            params=["storage"],
            start_date="2026-06-04T00:00:00Z",
            end_date="2026-06-05T00:00:00Z",
        )
        harvested = reaper.reap()

        assert isinstance(harvested, pd.DataFrame)
        assert harvested.empty

    @patch("cosecha.reaping.usace.tiny_retriever.fetch")
    def test_reap_api_error(self, mock_fetch):
        """Test error handling when fetch fails."""
        mock_fetch.side_effect = Exception("Connection timeout")

        reaper = ReservoirReaper(
            site_ids=["JPLT2"],
            params=["storage"],
            start_date="2026-06-04T00:00:00Z",
            end_date="2026-06-05T00:00:00Z",
        )

        with pytest.raises(APIError, match="Failed to fetch USACE"):
            reaper.reap()

    @patch("cosecha.reaping.usace.tiny_retriever.fetch")
    def test_reap_service_error(self, mock_fetch):
        """Test error handling when API returns HTTP error."""
        mock_fetch.side_effect = ServiceError("404 Not Found", "http://example.com")

        reaper = ReservoirReaper(
            site_ids=["JPLT2"],
            params=["storage"],
            start_date="2026-06-04T00:00:00Z",
            end_date="2026-06-05T00:00:00Z",
        )

        with pytest.raises(APIError, match="Failed to fetch USACE"):
            reaper.reap()

    @patch("cosecha.reaping.usace.tiny_retriever.fetch")
    def test_reap_partial_failure(self, mock_fetch):
        """Test that partial failures still return available data."""
        catalog = _make_catalog(["JPLT2"], ["storage", "elevation"])
        ts_response = [
            # First param returns data
            {
                "values": [["2026-06-04T01:00:00Z", 100.0]],
                "unit": "ac-ft",
            },
            # Second param returns empty
            {"values": []},
        ]
        mock_fetch.side_effect = [[catalog], ts_response]

        reaper = ReservoirReaper(
            site_ids=["JPLT2"],
            params=["storage", "elevation"],
            start_date="2026-06-04T00:00:00Z",
            end_date="2026-06-05T00:00:00Z",
        )
        harvested = reaper.reap()

        assert len(harvested) == 1
        assert harvested["variable"].iloc[0] == "storage"

    @patch("cosecha.reaping.usace.tiny_retriever.fetch")
    def test_reap_stores_data_on_instance(self, mock_fetch):
        """Test that reap() stores data on the instance."""
        catalog = _make_catalog(["JPLT2"], ["storage"])
        ts_response = [
            {
                "values": [["2026-06-04T01:00:00Z", 100.0]],
                "unit": "ac-ft",
            }
        ]
        mock_fetch.side_effect = [[catalog], ts_response]

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

    @patch("cosecha.reaping.usace.tiny_retriever.fetch")
    def test_reap_with_rename_columns(self, mock_fetch):
        """Test reap applies rename_columns transformation."""
        catalog = _make_catalog(["JPLT2"], ["storage"])
        ts_response = [
            {
                "values": [
                    ["2026-06-04T01:00:00Z", 150.0],
                    ["2026-06-04T02:00:00Z", 152.0],
                ],
                "unit": "ac-ft",
            }
        ]
        mock_fetch.side_effect = [[catalog], ts_response]

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

    @patch("cosecha.reaping.usace.tiny_retriever.fetch")
    def test_reap_with_unit_conversions(self, mock_fetch):
        """Test reap applies unit_conversions transformation."""
        catalog = _make_catalog(["JPLT2"], ["outflow"])
        ts_response = [
            {
                "values": [
                    ["2026-06-04T01:00:00Z", 100.0],
                    ["2026-06-04T02:00:00Z", 200.0],
                ],
                "unit": "cfs",
            }
        ]
        mock_fetch.side_effect = [[catalog], ts_response]

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

    @patch("cosecha.reaping.usace.tiny_retriever.fetch")
    def test_reap_with_filter_columns(self, mock_fetch):
        """Test reap applies filter_columns transformation."""
        catalog = _make_catalog(["JPLT2"], ["storage"])
        ts_response = [
            {
                "values": [
                    ["2026-06-04T01:00:00Z", 150.0],
                ],
                "unit": "ac-ft",
            }
        ]
        mock_fetch.side_effect = [[catalog], ts_response]

        reaper = ReservoirReaper(
            site_ids=["JPLT2"],
            params=["storage"],
            start_date="2026-06-04T00:00:00Z",
            end_date="2026-06-05T00:00:00Z",
            transformations={"filter_columns": ["site_id", "datetime", "value"]},
        )
        harvested = reaper.reap()

        assert list(harvested.columns) == ["site_id", "datetime", "value"]

    @patch("cosecha.reaping.usace.tiny_retriever.fetch")
    def test_reap_with_multiple_transformations(self, mock_fetch):
        """Test reap applies multiple transformations in sequence."""
        catalog = _make_catalog(["JPLT2"], ["outflow"])
        ts_response = [
            {
                "values": [
                    ["2026-06-04T01:00:00Z", 100.0],
                ],
                "unit": "cfs",
            }
        ]
        mock_fetch.side_effect = [[catalog], ts_response]

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

    @patch("cosecha.reaping.usace.tiny_retriever.fetch")
    def test_reap_no_transformations(self, mock_fetch):
        """Test reap with no transformations returns raw data."""
        catalog = _make_catalog(["JPLT2"], ["storage"])
        ts_response = [
            {
                "values": [
                    ["2026-06-04T01:00:00Z", 150.0],
                ],
                "unit": "ac-ft",
            }
        ]
        mock_fetch.side_effect = [[catalog], ts_response]

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

    @patch("cosecha.reaping.usace.tiny_retriever.fetch")
    def test_request_uses_correct_url(self, mock_fetch):
        """Test that the correct URL is constructed for the provider."""
        catalog = _make_catalog(["JPLT2"], ["storage"])
        mock_fetch.side_effect = [[catalog], [{"values": []}]]

        reaper = ReservoirReaper(
            site_ids=["JPLT2"],
            params=["storage"],
            start_date="2026-06-04T00:00:00Z",
            end_date="2026-06-05T00:00:00Z",
            provider="nwk",
        )
        reaper.reap()

        # Second call is the timeseries fetch
        call_args = mock_fetch.call_args_list[1]
        urls = call_args[0][0]
        assert all("nwk" in u for u in urls)

    @patch("cosecha.reaping.usace.tiny_retriever.fetch")
    def test_request_uses_correct_ts_name(self, mock_fetch):
        """Test that the correct time series name is sent."""
        catalog = [
            {
                "code": "JPLT2",
                "timeseries": [
                    {"tsid": "JPLT2.Flow-Out.Inst.1Hour.0.Rev-SWF-REGI", "label": "Outflow"}
                ],
            }
        ]
        mock_fetch.side_effect = [[catalog], [{"values": []}]]

        reaper = ReservoirReaper(
            site_ids=["JPLT2"],
            params=["outflow"],
            start_date="2026-06-04T00:00:00Z",
            end_date="2026-06-05T00:00:00Z",
        )
        reaper.reap()

        call_args = mock_fetch.call_args_list[1]
        urls = call_args[0][0]
        assert "JPLT2.Flow-Out.Inst.1Hour.0.Rev-SWF-REGI" in urls[0]

    @patch("cosecha.reaping.usace.tiny_retriever.fetch")
    def test_request_timeout(self, mock_fetch):
        """Test that requests are made with a timeout."""
        catalog = _make_catalog(["JPLT2"], ["storage"])
        mock_fetch.side_effect = [[catalog], [{"values": []}]]

        reaper = ReservoirReaper(
            site_ids=["JPLT2"],
            params=["storage"],
            start_date="2026-06-04T00:00:00Z",
            end_date="2026-06-05T00:00:00Z",
        )
        reaper.reap()

        # Second call is the timeseries fetch
        call_args = mock_fetch.call_args_list[1]
        timeout = call_args[1]["timeout"]
        assert timeout == 120

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
