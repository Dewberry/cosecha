"""Tests for base reaper classes."""

from __future__ import annotations

import boto3
import fsspec
import moto
import pandas as pd
import pytest
import s3fs
import xarray as xr

from cosecha.exceptions import ReaperError
from cosecha.reaping.base import GriddedReaper, TimeSeriesReaper


class _TSReaper(TimeSeriesReaper):
    def __init__(self):
        super().__init__()

    def _validate_params(self):
        pass

    def _reap(self):
        return pd.DataFrame({"a": [1, 2], "b": [3, 4]})


class _GridReaper(GriddedReaper):
    def __init__(self):
        super().__init__()

    def _validate_params(self):
        pass

    def _reap(self):
        return xr.Dataset({"temp": (["x", "y"], [[1.0, 2.0], [3.0, 4.0]])})


class TestEnsureData:
    """Test _ensure_data method on ReaperBase via a concrete subclass."""

    def test_data_is_none_raises(self):
        """Calling _ensure_data when data is None raises ReaperError."""
        reaper = _TSReaper()
        assert reaper.data is None
        with pytest.raises(ReaperError, match=r"No data to sow. Call reap\(\) first."):
            reaper._ensure_data(pd.DataFrame, "time-series (DataFrame)")

    def test_data_wrong_type_raises(self):
        """Calling _ensure_data with wrong data type raises ReaperError."""
        reaper = _TSReaper()
        reaper.data = "not a dataframe"
        with pytest.raises(ReaperError, match="Data is not time-series"):
            reaper._ensure_data(pd.DataFrame, "time-series (DataFrame)")

    def test_data_correct_type(self):
        """Calling _ensure_data with correct data type does not raise."""
        reaper = _TSReaper()
        reaper.data = pd.DataFrame({"x": [1]})
        reaper._ensure_data(pd.DataFrame, "time-series (DataFrame)")


class TestSowToParquet:
    """Test TimeSeriesReaper.sow_to_parquet."""

    def test_successful_write(self, tmp_path):
        """Parquet write creates file, returns path string, and is readable."""
        reaper = _TSReaper()
        reaper.reap()
        out = tmp_path / "output.parquet"

        result = reaper.sow_to_parquet(out)

        assert out.exists()
        assert isinstance(result, str)
        df = pd.read_parquet(out)
        pd.testing.assert_frame_equal(df, reaper.data)

    def test_no_data_raises(self, tmp_path):
        """sow_to_parquet raises ReaperError when data is None."""
        reaper = _TSReaper()
        with pytest.raises(ReaperError, match="No data to sow"):
            reaper.sow_to_parquet(tmp_path / "output.parquet")

    def test_creates_parent_directories(self, tmp_path):
        """sow_to_parquet creates parent directories that do not exist."""
        reaper = _TSReaper()
        reaper.reap()
        nested = tmp_path / "a" / "b" / "c" / "output.parquet"

        reaper.sow_to_parquet(nested)

        assert nested.exists()


class TestSowToZarr:
    """Test GriddedReaper.sow_to_zarr."""

    def test_successful_write(self, tmp_path):
        """Zarr write creates store and returns path string."""
        reaper = _GridReaper()
        reaper.reap()
        store_path = tmp_path / "output.zarr"

        result = reaper.sow_to_zarr(store_path)

        assert store_path.exists()
        assert isinstance(result, str)

    def test_no_data_raises(self, tmp_path):
        """sow_to_zarr raises ReaperError when data is None."""
        reaper = _GridReaper()
        with pytest.raises(ReaperError, match="No data to sow"):
            reaper.sow_to_zarr(tmp_path / "output.zarr")


class TestSowToS3:
    """Test S3 write support for parquet and zarr."""

    @pytest.fixture()
    def _mock_s3(self, monkeypatch):
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
        s3fs.S3FileSystem.clear_instance_cache()
        with moto.mock_aws():
            boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="test-bucket")
            yield
        s3fs.S3FileSystem.clear_instance_cache()

    def test_parquet_s3(self, _mock_s3):
        """sow_to_parquet round-trips data through a mock S3 bucket."""
        reaper = _TSReaper()
        reaper.reap()

        result = reaper.sow_to_parquet("s3://test-bucket/data.parquet")

        assert result == "s3://test-bucket/data.parquet"
        with fsspec.open("s3://test-bucket/data.parquet", "rb") as f:
            df = pd.read_parquet(f)
        pd.testing.assert_frame_equal(df, reaper.data)

    def test_zarr_s3(self, mocker):
        """sow_to_zarr passes S3 URI through to xarray without wrapping in Path."""
        reaper = _GridReaper()
        reaper.reap()
        mock_to_zarr = mocker.patch.object(type(reaper.data), "to_zarr")

        result = reaper.sow_to_zarr("s3://test-bucket/output.zarr")

        assert result == "s3://test-bucket/output.zarr"
        mock_to_zarr.assert_called_once_with(
            "s3://test-bucket/output.zarr", mode="w", consolidated=True
        )


class TestSowToNetcdf:
    """Test GriddedReaper.sow_to_netcdf."""

    def test_successful_write(self, tmp_path):
        """NetCDF write creates .nc file and returns path string."""
        reaper = _GridReaper()
        reaper.reap()
        nc_path = tmp_path / "output.nc"

        result = reaper.sow_to_netcdf(nc_path)

        assert nc_path.exists()
        assert isinstance(result, str)

    def test_no_data_raises(self, tmp_path):
        """sow_to_netcdf raises ReaperError when data is None."""
        reaper = _GridReaper()
        with pytest.raises(ReaperError, match="No data to sow"):
            reaper.sow_to_netcdf(tmp_path / "output.nc")


class TestSowToIceberg:
    """Test TimeSeriesReaper.sow_to_iceberg."""

    def test_warehouse_path_is_file_raises(self, tmp_path):
        """sow_to_iceberg raises ReaperError when warehouse_path is a file."""
        reaper = _TSReaper()
        reaper.reap()
        file_path = tmp_path / "not_a_dir"
        file_path.write_text("I am a file")

        with pytest.raises(ReaperError, match="warehouse_path must be a directory, got file"):
            reaper.sow_to_iceberg(file_path, table_name="test_table")

    def test_no_data_raises(self, tmp_path):
        """sow_to_iceberg raises ReaperError when data is None."""
        reaper = _TSReaper()
        with pytest.raises(ReaperError, match="No data to sow"):
            reaper.sow_to_iceberg(tmp_path / "warehouse", table_name="test_table")

    def test_successful_write(self, tmp_path):
        """sow_to_iceberg writes data and returns fully qualified table name."""
        reaper = _TSReaper()
        reaper.reap()
        warehouse = tmp_path / "warehouse"

        result = reaper.sow_to_iceberg(warehouse, table_name="my_table")

        assert result == "default.my_table"
        assert warehouse.exists()


class TestSowToIcechunk:
    """Test GriddedReaper.sow_to_icechunk."""

    def test_storage_path_is_file_raises(self, tmp_path):
        """sow_to_icechunk raises ReaperError when storage_path is a file."""
        reaper = _GridReaper()
        reaper.reap()
        file_path = tmp_path / "not_a_dir"
        file_path.write_text("I am a file")

        with pytest.raises(ReaperError, match="storage_path must be a directory, got file"):
            reaper.sow_to_icechunk(file_path, group_path="test_group")

    def test_no_data_raises(self, tmp_path):
        """sow_to_icechunk raises ReaperError when data is None."""
        reaper = _GridReaper()
        with pytest.raises(ReaperError, match="No data to sow"):
            reaper.sow_to_icechunk(tmp_path / "store", group_path="test_group")

    def test_successful_write(self, tmp_path):
        """sow_to_icechunk writes data and returns path string."""
        reaper = _GridReaper()
        reaper.reap()
        store_path = tmp_path / "icechunk_store"

        result = reaper.sow_to_icechunk(store_path, group_path="my_group")

        assert isinstance(result, str)
        assert store_path.exists()
