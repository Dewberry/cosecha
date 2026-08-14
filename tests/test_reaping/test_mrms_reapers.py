"""Tests for MRMS reaper implementation."""

from __future__ import annotations

import gzip
from io import BytesIO
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from cosecha.exceptions import APIError, DateRangeError, ReaperError
from cosecha.reaping.mrms import MRMSReaper


class TestMRMSReaper:
    """Test MRMSReaper implementation."""

    def test_initialization_single_time(self):
        """Test valid initialization with single time."""
        reaper = MRMSReaper(dates=("2026-01-01", "2026-01-01"))
        assert reaper.variable == "MultiSensor_QPE_01H_Pass2_00.00"
        assert reaper.start_date == pd.Timestamp("2026-01-01", tz="UTC")
        assert reaper.end_date == pd.Timestamp("2026-01-01", tz="UTC")

    def test_initialization_latest_time(self):
        """Test valid initialization with latest time."""
        reaper = MRMSReaper(dates="latest")
        assert reaper.variable == "MultiSensor_QPE_01H_Pass2_00.00"
        assert reaper.is_latest is True

    def test_initialization_time_range(self):
        """Test valid initialization with time range."""
        reaper = MRMSReaper(dates=("2026-01-01", "2026-01-02"))
        assert reaper.start_date == pd.Timestamp("2026-01-01", tz="UTC")
        assert reaper.end_date == pd.Timestamp("2026-01-02", tz="UTC")

    def test_initialization_custom_variable(self):
        """Test valid initialization with custom variable."""
        reaper = MRMSReaper(dates=("2026-01-01", "2026-01-01"), variable="custom_var")
        assert reaper.variable == "custom_var"

    def test_invalid_no_time(self):
        """Test initialization fails with no time parameters."""
        with pytest.raises(TypeError, match="missing 1 required positional argument: 'dates'"):
            MRMSReaper()

    def test_invalid_start_after_end(self):
        """Test initialization fails when start_time > end_time."""
        with pytest.raises(DateRangeError, match="must be <="):
            MRMSReaper(dates=("2026-01-02", "2026-01-01"))

    def test_validate_params_valid(self):
        """Test _validate_params with valid parameters."""
        reaper = MRMSReaper(dates=("2026-01-01", "2026-01-01"))
        reaper._validate_params()  # Should not raise

    def test_find_available_files_no_files(self, mocker):
        """Test _find_available_files raises APIError when no files are found."""
        reaper = MRMSReaper(dates=("2026-01-01 12:00", "2026-01-01 12:00"))
        mock_s3 = mocker.MagicMock()
        mock_s3.ls.return_value = []
        reaper.aws = mock_s3

        with pytest.raises(APIError, match="No files found for"):
            reaper._find_available_files()

    def test_reap_mocked_success(self, mocker):
        """Test reap with mocked _fetch_data."""
        mock_ds = xr.Dataset(
            {
                "test_var": (("time", "latitude", "longitude"), [[[1, 2], [3, 4]]]),
            }
        )

        reaper = MRMSReaper(dates=("2026-01-01", "2026-01-01"))
        mocker.patch.object(reaper, "_fetch_data", return_value=mock_ds)

        harvested = reaper.reap()

        assert isinstance(harvested, xr.Dataset)
        assert len(harvested.data_vars) == 1
        assert "test_var" in harvested.data_vars

    def test_reap_api_error_handling(self, mocker):
        """Test that reap handles errors from _fetch_data gracefully."""
        reaper = MRMSReaper(dates=("2026-01-01", "2026-01-01"))
        mocker.patch.object(reaper, "_fetch_data", side_effect=APIError("S3 fetch failed"))

        with pytest.raises(ReaperError, match="MRMS reaping failed"):
            reaper.reap()

    def test_invalid_time_string(self):
        """Test that an unparsable time string raises DateRangeError."""
        with pytest.raises(DateRangeError, match="Could not parse date"):
            MRMSReaper(dates=("not-a-valid-date", "not-a-valid-date"))

    def test_find_available_files_matches_hour(self, mocker):
        """Test _find_available_files returns only files matching the requested hour."""
        reaper = MRMSReaper(dates=("2026-01-01 12:00", "2026-01-01 12:00"))
        mock_s3 = mocker.MagicMock()
        base = "noaa-mrms-pds/CONUS/MultiSensor_QPE_01H_Pass2_00.00/20260101"
        mock_s3.ls.return_value = [
            f"{base}/MRMS_MultiSensor_QPE_01H_Pass2_00.00_20260101-120000.grib2.gz",
            f"{base}/MRMS_MultiSensor_QPE_01H_Pass2_00.00_20260101-130000.grib2.gz",
            f"{base}/MRMS_MultiSensor_QPE_01H_Pass2_00.00_20260101-110000.grib2.gz",
        ]
        reaper.aws = mock_s3

        result = reaper._find_available_files()

        assert len(result) == 1
        assert "120000" in result[0]

    def test_fetch_data_single_time(self, mocker):
        """Test _fetch_data with a single time returns the dataset directly."""
        reaper = MRMSReaper(dates=("2026-01-01 12:00", "2026-01-01 12:00"))
        single_ds = xr.Dataset(
            {"precip": (("time", "latitude", "longitude"), [[[1.0, 2.0], [3.0, 4.0]]])}
        )
        mocker.patch.object(
            reaper,
            "_find_available_files",
            return_value=["noaa-mrms-pds/CONUS/var/20260101/file_20260101-120000.grib2.gz"],
        )
        mocker.patch.object(reaper, "_process_single_file", return_value=single_ds)

        result = reaper._fetch_data()

        assert isinstance(result, xr.Dataset)
        assert "precip" in result.data_vars

    def test_fetch_data_time_range(self, mocker):
        """Test _fetch_data with a time range concatenates multiple datasets."""
        reaper = MRMSReaper(dates=("2026-01-01 12:00", "2026-01-01 14:00"))

        ds1 = xr.Dataset(
            {"precip": (("time", "latitude", "longitude"), [[[1.0, 2.0], [3.0, 4.0]]])},
            coords={"time": [pd.Timestamp("2026-01-01 12:00")]},
        )
        ds2 = xr.Dataset(
            {"precip": (("time", "latitude", "longitude"), [[[5.0, 6.0], [7.0, 8.0]]])},
            coords={"time": [pd.Timestamp("2026-01-01 13:00")]},
        )
        ds3 = xr.Dataset(
            {
                "precip": (
                    ("time", "latitude", "longitude"),
                    [[[9.0, 10.0], [11.0, 12.0]]],
                )
            },
            coords={"time": [pd.Timestamp("2026-01-01 14:00")]},
        )

        mocker.patch.object(
            reaper,
            "_find_available_files",
            return_value=[
                "noaa-mrms-pds/CONUS/var/20260101/file_20260101-120000.grib2.gz",
                "noaa-mrms-pds/CONUS/var/20260101/file_20260101-130000.grib2.gz",
                "noaa-mrms-pds/CONUS/var/20260101/file_20260101-140000.grib2.gz",
            ],
        )
        mocker.patch.object(reaper, "_process_single_file", side_effect=[ds1, ds2, ds3])

        result = reaper._fetch_data()

        assert isinstance(result, xr.Dataset)
        assert result.sizes["time"] == 3

    def test_fetch_data_all_files_fail(self, mocker):
        """Test _fetch_data raises APIError when all files fail to process."""
        reaper = MRMSReaper(dates=("2026-01-01 12:00", "2026-01-01 12:00"))
        mocker.patch.object(
            reaper,
            "_find_available_files",
            return_value=[
                "noaa-mrms-pds/CONUS/var/20260101/file_20260101-120000.grib2.gz",
                "noaa-mrms-pds/CONUS/var/20260101/file_20260101-120100.grib2.gz",
            ],
        )
        mocker.patch.object(reaper, "_process_single_file", return_value=None)

        with pytest.raises(APIError, match="No data could be successfully fetched"):
            reaper._fetch_data()

    def test_reap_with_transformations(self, mocker):
        """Test _reap applies transformations when provided."""
        mock_ds = xr.Dataset({"old": (("time", "latitude", "longitude"), [[[1, 2], [3, 4]]])})

        reaper = MRMSReaper(
            dates=("2026-01-01", "2026-01-01"),
            transformations={"variable_rename": {"old": "new"}},
        )
        mocker.patch.object(reaper, "_fetch_data", return_value=mock_ds)

        result = reaper.reap()

        assert isinstance(result, xr.Dataset)
        assert "new" in result.data_vars
        assert "old" not in result.data_vars

    def test_process_single_file_retry_then_success(self, mocker):
        """Test _process_single_file succeeds after initial failure with retry."""
        reaper = MRMSReaper(dates=("2026-01-01 12:00", "2026-01-01 12:00"))
        mocker.patch("cosecha.reaping.mrms.time_mod.sleep")

        compressed = gzip.compress(b"fake grib2 data")

        call_count = 0

        def mock_open_side_effect(path, mode):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OSError("Connection reset")
            cm = MagicMock()
            cm.__enter__ = lambda s: BytesIO(compressed)
            cm.__exit__ = lambda s, *a: None
            return cm

        reaper.aws = MagicMock()
        reaper.aws.open.side_effect = mock_open_side_effect

        mock_da = xr.DataArray(
            np.ones((2, 3)),
            dims=["latitude", "longitude"],
            coords={
                "latitude": [40.0, 41.0],
                "longitude": [260.0, 261.0, 262.0],
            },
        )
        mocker.patch("xarray.load_dataarray", return_value=mock_da)

        file_path = (
            "noaa-mrms-pds/CONUS/MultiSensor_QPE_01H_Pass2_00.00/20260101/"
            "MRMS_MultiSensor_QPE_01H_Pass2_00.00_20260101-120000.grib2.gz"
        )
        result = reaper._process_single_file(file_path)

        assert result is not None
        assert isinstance(result, xr.Dataset)
        assert reaper.aws.open.call_count == 2

    def test_process_single_file_all_retries_fail(self, mocker):
        """Test _process_single_file returns None when all retries fail."""
        reaper = MRMSReaper(dates=("2026-01-01 12:00", "2026-01-01 12:00"))
        mocker.patch("cosecha.reaping.mrms.time_mod.sleep")

        reaper.aws = MagicMock()
        reaper.aws.open.side_effect = OSError("Connection refused")

        file_path = (
            "noaa-mrms-pds/CONUS/MultiSensor_QPE_01H_Pass2_00.00/20260101/"
            "MRMS_MultiSensor_QPE_01H_Pass2_00.00_20260101-120000.grib2.gz"
        )
        result = reaper._process_single_file(file_path)

        assert result is None
        assert reaper.aws.open.call_count == 3

    @pytest.mark.network
    def test_reap_network(self):
        """Test live MRMS fetch from S3."""
        reaper = MRMSReaper(dates=("2024-01-01 12:00", "2024-01-01 12:00"))
        harvested = reaper.reap()
        assert isinstance(harvested, xr.Dataset)
        assert len(harvested.data_vars) > 0
        assert "latitude" in harvested.coords
        assert "longitude" in harvested.coords
        assert harvested.longitude.values.min() >= -180
        assert harvested.longitude.values.max() <= 180
