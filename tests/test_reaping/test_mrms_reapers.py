"""Tests for MRMS reaper implementation."""

from __future__ import annotations

from datetime import datetime

import pytest
import xarray as xr

from cosecha.exceptions import APIError, DateRangeError, ReaperError
from cosecha.reaping.mrms import MRMSReaper


class TestMRMSReaper:
    """Test MRMSReaper implementation."""

    def test_initialization_single_time(self):
        """Test valid initialization with single time."""
        dt = datetime(2026, 1, 1)
        reaper = MRMSReaper(time=dt)
        assert reaper.variable == "MultiSensor_QPE_01H_Pass2_00.00"
        assert reaper.time == dt
        assert reaper.start_time is None
        assert reaper.end_time is None

    def test_initialization_time_range(self):
        """Test valid initialization with time range."""
        start = datetime(2026, 1, 1)
        end = datetime(2026, 1, 2)
        reaper = MRMSReaper(start_time=start, end_time=end)
        assert reaper.start_time == start
        assert reaper.end_time == end
        assert reaper.time is None

    def test_initialization_custom_variable(self):
        """Test valid initialization with custom variable."""
        reaper = MRMSReaper(time=datetime(2026, 1, 1), variable="custom_var")
        assert reaper.variable == "custom_var"

    def test_invalid_no_time(self):
        """Test initialization fails with no time parameters."""
        with pytest.raises(DateRangeError, match="Must provide either 'time' or both"):
            MRMSReaper()

    def test_invalid_start_after_end(self):
        """Test initialization fails when start_time > end_time."""
        start = datetime(2026, 1, 2)
        end = datetime(2026, 1, 1)
        with pytest.raises(DateRangeError, match="start_time must be <= end_time"):
            MRMSReaper(start_time=start, end_time=end)

    def test_validate_params_valid(self):
        """Test _validate_params with valid parameters."""
        reaper = MRMSReaper(time=datetime(2026, 1, 1))
        reaper._validate_params()  # Should not raise

    def test_find_available_files_no_files(self, mocker):
        """Test _find_available_files raises APIError when no files are found."""
        reaper = MRMSReaper(time=datetime(2026, 1, 1, 12))
        mock_s3 = mocker.MagicMock()
        mock_s3.ls.return_value = []
        reaper.aws = mock_s3

        with pytest.raises(APIError, match="No files found for"):
            reaper._find_available_files([datetime(2026, 1, 1, 12)])

    def test_reap_mocked_success(self, mocker):
        """Test reap with mocked fetch_data."""
        mock_ds = xr.Dataset(
            {
                "test_var": (("time", "latitude", "longitude"), [[[1, 2], [3, 4]]]),
            }
        )

        reaper = MRMSReaper(time=datetime(2026, 1, 1))
        mocker.patch.object(reaper, "fetch_data", return_value=mock_ds)

        harvested = reaper.reap()

        assert isinstance(harvested, xr.Dataset)
        assert len(harvested.data_vars) == 1
        assert "test_var" in harvested.data_vars

    def test_reap_api_error_handling(self, mocker):
        """Test that reap handles errors from fetch_data gracefully."""
        reaper = MRMSReaper(time=datetime(2026, 1, 1))
        mocker.patch.object(reaper, "fetch_data", side_effect=APIError("S3 fetch failed"))

        with pytest.raises(ReaperError, match="MRMS reaping failed"):
            reaper.reap()
