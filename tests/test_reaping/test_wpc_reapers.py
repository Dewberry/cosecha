"""Tests for WPC QPF reaper implementation."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from cosecha.exceptions import APIError, DataNotFoundError, DateRangeError, ReaperError
from cosecha.reaping.wpc import WPCQPFReaper

BASE_URL = "https://ftp-wpc.ncep.noaa.gov/2p5km_qpf/"


def _listing(files: list[str]) -> str:
    """Build a minimal Apache directory listing like the WPC server returns."""
    rows = "\n".join(f'<tr><td><a href="{f}">{f}</a></td></tr>' for f in files)
    return f"<html><body><table>\n{rows}\n</table></body></html>"


def _mock_qpf_dataset(hour: int) -> xr.Dataset:
    """Build a small dataset shaped like a decoded WPC QPF GRIB2 file."""
    init = pd.Timestamp("2026-09-23 12:00")
    return xr.Dataset(
        {"tp": (("step", "y", "x"), np.full((1, 2, 2), float(hour), dtype="float32"))},
        coords={
            "step": [pd.Timedelta(hours=hour)],
            "valid_time": ("step", [init + pd.Timedelta(hours=hour)]),
            "time": init,
            "latitude": (("y", "x"), [[30.0, 30.0], [31.0, 31.0]]),
            "longitude": (("y", "x"), [[265.0, 266.0], [265.0, 266.0]]),
        },
    )


class TestWPCQPFReaper:
    """Test WPCQPFReaper implementation."""

    def test_initialization_latest(self):
        """Test valid initialization with latest init_time."""
        reaper = WPCQPFReaper(init_time="latest")
        assert reaper.is_latest is True
        assert reaper.init_time is None
        assert reaper.forecast_hours is None

    def test_initialization_init_time(self):
        """Test valid initialization with an explicit init_time and range."""
        reaper = WPCQPFReaper(init_time="2026-09-23 12:00", forecast_hours=range(6, 25, 6))
        assert reaper.is_latest is False
        assert reaper.init_time == pd.Timestamp("2026-09-23 12:00", tz="UTC")
        assert reaper.forecast_hours == [6, 12, 18, 24]

    def test_invalid_time_string(self):
        """Test that an unparsable init_time raises DateRangeError."""
        with pytest.raises(DateRangeError, match="Could not parse date"):
            WPCQPFReaper(init_time="not-a-valid-date")

    def test_invalid_issuance_hour(self):
        """Test that an init_time outside the WPC issuance hours raises DateRangeError."""
        with pytest.raises(DateRangeError, match="00, 06, 12 or 18 UTC"):
            WPCQPFReaper(init_time="2026-09-23 03:00")

    @pytest.mark.parametrize("forecast_hours", [[], [0], [5], [180]])
    def test_invalid_forecast_hours(self, forecast_hours):
        """Test that invalid forecast_hours raise DateRangeError."""
        with pytest.raises(DateRangeError, match="positive multiples of 6"):
            WPCQPFReaper(init_time="latest", forecast_hours=forecast_hours)

    @patch("cosecha.reaping.wpc.tiny_retriever.fetch")
    def test_find_available_files_init_time(self, mock_fetch):
        """Test _find_available_files returns URLs for the requested init_time and hours."""
        mock_fetch.return_value = _listing(
            [
                "p06m_2026092312f006.grb",
                "p06m_2026092312f012.grb",
                "p06m_2026092312f018.grb",
                "p06m_2026092318f006.grb",
                "p24m_2026092312f024.grb",
            ]
        )
        reaper = WPCQPFReaper(init_time="2026-09-23 12:00", forecast_hours=[6, 12])

        result = reaper._find_available_files()

        assert result == [
            f"{BASE_URL}p06m_2026092312f006.grb",
            f"{BASE_URL}p06m_2026092312f012.grb",
        ]

    @patch("cosecha.reaping.wpc.tiny_retriever.fetch")
    def test_find_available_files_latest(self, mock_fetch):
        """Test latest selects the newest issuance with all requested hours."""
        mock_fetch.return_value = _listing(
            [
                "p06m_2026092312f006.grb",
                "p06m_2026092312f012.grb",
                "p06m_2026092318f012.grb",
            ]
        )
        reaper = WPCQPFReaper(init_time="latest", forecast_hours=[6, 12])

        result = reaper._find_available_files()

        assert reaper.init_time == pd.Timestamp("2026-09-23 12:00", tz="UTC")
        assert len(result) == 2

    @patch("cosecha.reaping.wpc.tiny_retriever.fetch")
    def test_find_available_files_missing_hours(self, mock_fetch, mocker):
        """Test _find_available_files skips and warns about missing forecast hours."""
        mock_fetch.return_value = _listing(["p06m_2026092312f006.grb"])
        mock_logger = mocker.patch("cosecha.reaping.wpc.logger")
        reaper = WPCQPFReaper(init_time="2026-09-23 12:00", forecast_hours=[6, 12])

        result = reaper._find_available_files()

        assert result == [f"{BASE_URL}p06m_2026092312f006.grb"]
        mock_logger.warning.assert_called_once()

    @patch("cosecha.reaping.wpc.tiny_retriever.fetch")
    def test_find_available_files_no_files(self, mock_fetch):
        """Test _find_available_files raises DataNotFoundError when nothing matches."""
        mock_fetch.return_value = _listing(["p06m_2026092312f006.grb"])
        reaper = WPCQPFReaper(init_time="2026-09-01 12:00")

        with pytest.raises(DataNotFoundError, match="No WPC QPF files found"):
            reaper._find_available_files()

    @patch("cosecha.reaping.wpc.tiny_retriever.fetch")
    def test_find_available_files_api_error(self, mock_fetch):
        """Test _find_available_files wraps listing failures in APIError."""
        mock_fetch.side_effect = Exception("Connection failed")
        reaper = WPCQPFReaper(init_time="latest")

        with pytest.raises(APIError, match="Could not list available WPC QPF files"):
            reaper._find_available_files()

    @patch("cosecha.reaping.wpc.tiny_retriever.download")
    def test_fetch_data_multiple_steps(self, mock_download, mocker):
        """Test _fetch_data concatenates files along step."""
        reaper = WPCQPFReaper(init_time="2026-09-23 12:00", forecast_hours=[6, 12])
        mocker.patch.object(
            reaper,
            "_find_available_files",
            return_value=[
                f"{BASE_URL}p06m_2026092312f006.grb",
                f"{BASE_URL}p06m_2026092312f012.grb",
            ],
        )
        mocker.patch.object(
            reaper,
            "_process_single_file",
            side_effect=[_mock_qpf_dataset(6), _mock_qpf_dataset(12)],
        )

        result = reaper._fetch_data()

        assert isinstance(result, xr.Dataset)
        assert result.sizes["step"] == 2
        assert result["latitude"].dims == ("y", "x")
        assert result["longitude"].values.max() <= 180
        mock_download.assert_called_once()

    @patch("cosecha.reaping.wpc.tiny_retriever.download")
    def test_fetch_data_download_error(self, mock_download, mocker):
        """Test _fetch_data wraps download failures in APIError."""
        mock_download.side_effect = Exception("Timeout")
        reaper = WPCQPFReaper(init_time="2026-09-23 12:00", forecast_hours=[6])
        mocker.patch.object(
            reaper,
            "_find_available_files",
            return_value=[f"{BASE_URL}p06m_2026092312f006.grb"],
        )

        with pytest.raises(APIError, match="Could not download WPC QPF files"):
            reaper._fetch_data()

    def test_reap_mocked_success(self, mocker):
        """Test reap with mocked _fetch_data."""
        reaper = WPCQPFReaper(init_time="latest")
        mocker.patch.object(reaper, "_fetch_data", return_value=_mock_qpf_dataset(6))

        harvested = reaper.reap()

        assert isinstance(harvested, xr.Dataset)
        assert "tp" in harvested.data_vars

    def test_reap_api_error_handling(self, mocker):
        """Test that reap wraps errors from _fetch_data."""
        reaper = WPCQPFReaper(init_time="latest")
        mocker.patch.object(reaper, "_fetch_data", side_effect=APIError("fetch failed"))

        with pytest.raises(ReaperError, match="WPC QPF reaping failed"):
            reaper.reap()

    def test_reap_with_transformations(self, mocker):
        """Test _reap applies transformations when provided."""
        reaper = WPCQPFReaper(
            init_time="latest",
            transformations={"variable_rename": {"tp": "qpf"}},
        )
        mocker.patch.object(reaper, "_fetch_data", return_value=_mock_qpf_dataset(6))

        result = reaper.reap()

        assert "qpf" in result.data_vars
        assert "tp" not in result.data_vars

    @pytest.mark.network
    def test_reap_network(self):
        """Test live WPC QPF fetch."""
        reaper = WPCQPFReaper(init_time="latest", forecast_hours=[6])
        harvested = reaper.reap()
        assert isinstance(harvested, xr.Dataset)
        assert "tp" in harvested.data_vars
        assert "latitude" in harvested.coords
        assert "longitude" in harvested.coords
        assert harvested.longitude.values.min() >= -180
        assert harvested.longitude.values.max() <= 180
