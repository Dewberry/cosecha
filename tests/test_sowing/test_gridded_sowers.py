"""Tests for gridded sower implementations (Zarr, IceChunk, NetCDF)."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pytest
import xarray as xr

from cosecha.data_models import HarvestedData
from cosecha.reaping.utils import apply_gridded_transformations
from cosecha.sowing.exceptions import SowerError
from cosecha.sowing.icechunk import IceChunkSower
from cosecha.sowing.netcdf import NetCDFSower
from cosecha.sowing.zarr import ZarrSower


class TestZarrSower:
    """Test ZarrSower implementation."""

    @pytest.fixture
    def temp_output_dir(self):
        """Temporary directory for output."""
        with TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_dataset(self):
        """Sample xarray Dataset with gridded data."""
        return xr.Dataset(
            {
                "precip": (("y", "x"), [[1.0, 2.0], [3.0, 4.0]]),
                "temp": (("y", "x"), [[20.0, 21.0], [19.0, 22.0]]),
            }
        )

    @pytest.fixture
    def harvested_data(self, sample_dataset):
        """Create HarvestedData with xarray Dataset."""
        return HarvestedData(
            data=sample_dataset,
            source_name="HRRR Forecast",
            timestamp=__import__("pandas").Timestamp("2026-01-01 12:00:00"),
            variable_names=["precip", "temp"],
            metadata={"model": "HRRR", "init_time": "2026-01-01 00:00"},
        )

    def test_initialization(self, temp_output_dir):
        """Test ZarrSower initialization."""
        sower = ZarrSower(output_dir=temp_output_dir)
        assert sower.output_dir.exists()
        assert sower.output_dir.is_dir()

    def test_initialization_creates_directory(self, temp_output_dir):
        """Test that initialization creates directory if needed."""
        new_dir = temp_output_dir / "zarr" / "store"
        ZarrSower(output_dir=new_dir)
        assert new_dir.exists()

    def test_initialization_fails_with_file(self, temp_output_dir):
        """Test initialization fails if output_dir is a file."""
        file_path = temp_output_dir / "file.txt"
        file_path.touch()

        with pytest.raises(ValueError, match="must be a directory"):
            ZarrSower(output_dir=file_path)

    def test_validate_input_dataset(self, temp_output_dir, harvested_data):
        """Test validation passes for Dataset."""
        sower = ZarrSower(output_dir=temp_output_dir)
        sower._validate_input(harvested_data)  # Should not raise

    def test_validate_input_timeseries_fails(self, temp_output_dir):
        """Test validation fails for time-series DataFrame data."""
        sower = ZarrSower(output_dir=temp_output_dir)
        df = pd.DataFrame(
            {"time": pd.date_range("2026-01-01", periods=5), "value": [1.0, 2.0, 3.0, 4.0, 5.0]}
        )

        data = HarvestedData(
            data=df,
            source_name="USGS Streamflow",
            timestamp=pd.Timestamp("2026-01-01"),
            variable_names=["discharge"],
            metadata={"site_id": "01650000"},
        )

        with pytest.raises(SowerError, match="only supports gridded"):
            sower._validate_input(data)

    def test_sow_creates_zarr_store(self, temp_output_dir, harvested_data):
        """Test that sow creates a Zarr store."""
        path = harvested_data.sow_to_zarr(output_dir=temp_output_dir)

        assert Path(path).exists()
        assert path.endswith(".zarr")

    def test_sow_returns_path_string(self, temp_output_dir, harvested_data):
        """Test that sow returns a string path."""
        path = harvested_data.sow_to_zarr(output_dir=temp_output_dir)
        assert isinstance(path, str)

    def test_sow_with_transformations(self, temp_output_dir, harvested_data):
        """Test that transformations can be applied manually before sowing."""
        transformations = {"unit_conversions": {"temp": 1.8}}
        transformed_data = apply_gridded_transformations(harvested_data, transformations)
        
        path = transformed_data.sow_to_zarr(output_dir=temp_output_dir)
        
        assert Path(path).exists()
        
        # Verify transformation (temp should be multiplied by 1.8)
        ds = xr.open_zarr(path)
        assert ds["temp"].values[0, 0] == 20.0 * 1.8  # original was 20.0

class TestIceChunkSower:
    """Test IceChunkSower implementation."""

    @pytest.fixture
    def temp_storage_dir(self):
        """Temporary directory for IceChunk storage."""
        with TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_dataset(self):
        """Sample xarray Dataset with gridded data."""
        return xr.Dataset(
            {
                "precip": (("y", "x"), [[1.0, 2.0], [3.0, 4.0]]),
                "temp": (("y", "x"), [[20.0, 21.0], [19.0, 22.0]]),
            }
        )

    @pytest.fixture
    def harvested_data(self, sample_dataset):
        """Create HarvestedData with xarray Dataset."""
        return HarvestedData(
            data=sample_dataset,
            source_name="HRRR Forecast",
            timestamp=pd.Timestamp("2026-01-01 12:00:00"),
            variable_names=["precip", "temp"],
            metadata={"model": "HRRR", "init_time": "2026-01-01 00:00"},
        )

    def test_initialization(self, temp_storage_dir):
        """Test IceChunkSower initialization."""
        sower = IceChunkSower(storage_path=temp_storage_dir)
        assert sower.storage_path.exists()
        assert sower.storage_path.is_dir()

    def test_initialization_creates_directory(self, temp_storage_dir):
        """Test that initialization creates directory if needed."""
        new_dir = temp_storage_dir / "ice" / "chunk" / "store"
        IceChunkSower(storage_path=new_dir)
        assert new_dir.exists()

    def test_initialization_fails_with_file(self, temp_storage_dir):
        """Test initialization fails if storage_path is a file."""
        file_path = temp_storage_dir / "file.txt"
        file_path.touch()

        with pytest.raises(ValueError, match="must be a directory"):
            IceChunkSower(storage_path=file_path)

    def test_validate_input_dataset(self, temp_storage_dir, harvested_data):
        """Test validation passes for Dataset."""
        sower = IceChunkSower(storage_path=temp_storage_dir)
        sower._validate_input(harvested_data)  # Should not raise

    def test_validate_input_timeseries_fails(self, temp_storage_dir):
        """Test validation fails for time-series DataFrame data."""
        sower = IceChunkSower(storage_path=temp_storage_dir)
        df = pd.DataFrame(
            {"time": pd.date_range("2026-01-01", periods=5), "value": [1.0, 2.0, 3.0, 4.0, 5.0]}
        )

        data = HarvestedData(
            data=df,
            source_name="USGS Streamflow",
            timestamp=pd.Timestamp("2026-01-01"),
            variable_names=["discharge"],
            metadata={"site_id": "01650000"},
        )

        with pytest.raises(SowerError, match="only supports gridded"):
            sower._validate_input(data)


    def test_sow_creates_icechunk_store(self, temp_storage_dir, harvested_data):
        """Test that sow creates an IceChunk store."""
        harvested_data.sow_to_icechunk(storage_path=temp_storage_dir)

        # IceChunk manages its own internal layout, so we check if the storage root exists
        # and has populated data rather than assuming a Zarr v2 style literal directory path
        assert temp_storage_dir.exists()
        assert any(temp_storage_dir.iterdir())

    def test_sow_returns_path_string(self, temp_storage_dir, harvested_data):
        """Test that sow returns a string path."""
        path = harvested_data.sow_to_icechunk(storage_path=temp_storage_dir)
        assert isinstance(path, str)


@pytest.mark.requires_netcdf
class TestNetCDFSower:
    """Test NetCDFSower implementation."""

    @pytest.fixture
    def temp_output_dir(self):
        """Temporary directory for output."""
        with TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_dataset(self):
        """Sample xarray Dataset with gridded data."""
        return xr.Dataset(
            {
                "precip": (("y", "x"), [[1.0, 2.0], [3.0, 4.0]]),
                "temp": (("y", "x"), [[20.0, 21.0], [19.0, 22.0]]),
            }
        )

    @pytest.fixture
    def harvested_data(self, sample_dataset):
        """Create HarvestedData with xarray Dataset."""
        return HarvestedData(
            data=sample_dataset,
            source_name="HRRR Forecast",
            timestamp=__import__("pandas").Timestamp("2026-01-01 12:00:00"),
            variable_names=["precip", "temp"],
            metadata={"model": "HRRR", "init_time": "2026-01-01 00:00"},
        )

    def test_initialization(self, temp_output_dir):
        """Test NetCDFSower initialization."""
        sower = NetCDFSower(output_dir=temp_output_dir)
        assert sower.output_dir.exists()
        assert sower.output_dir.is_dir()

    def test_initialization_creates_directory(self, temp_output_dir):
        """Test that initialization creates directory if needed."""
        new_dir = temp_output_dir / "netcdf" / "output"
        NetCDFSower(output_dir=new_dir)
        assert new_dir.exists()

    def test_initialization_with_compression(self, temp_output_dir):
        """Test initialization with custom compression settings."""
        sower = NetCDFSower(output_dir=temp_output_dir, compression="zlib", compression_level=7)
        assert sower.compression == "zlib"
        assert sower.compression_level == 7

    def test_initialization_fails_with_file(self, temp_output_dir):
        """Test initialization fails if output_dir is a file."""
        file_path = temp_output_dir / "file.txt"
        file_path.touch()

        with pytest.raises(ValueError, match="must be a directory"):
            NetCDFSower(output_dir=file_path)

    def test_initialization_invalid_compression(self, temp_output_dir):
        """Test initialization fails with invalid compression."""
        with pytest.raises(ValueError, match="compression must be"):
            NetCDFSower(output_dir=temp_output_dir, compression="invalid")

    def test_validate_input_dataset(self, temp_output_dir, harvested_data):
        """Test validation passes for Dataset."""
        sower = NetCDFSower(output_dir=temp_output_dir)
        sower._validate_input(harvested_data)  # Should not raise

    def test_validate_input_timeseries_fails(self, temp_output_dir):
        """Test validation fails for time-series DataFrame data."""
        sower = NetCDFSower(output_dir=temp_output_dir)
        df = pd.DataFrame(
            {"time": pd.date_range("2026-01-01", periods=5), "value": [1.0, 2.0, 3.0, 4.0, 5.0]}
        )

        data = HarvestedData(
            data=df,
            source_name="USGS Streamflow",
            timestamp=pd.Timestamp("2026-01-01"),
            variable_names=["discharge"],
            metadata={"site_id": "01650000"},
        )

        with pytest.raises(SowerError, match="only supports gridded"):
            sower._validate_input(data)

    def test_sow_creates_netcdf_file(self, temp_output_dir, harvested_data):
        """Test that sow creates a NetCDF file."""
        path = harvested_data.sow_to_netcdf(output_dir=temp_output_dir)

        assert Path(path).exists()
        assert path.endswith(".nc")

    def test_sow_returns_path_string(self, temp_output_dir, harvested_data):
        """Test that sow returns a string path."""
        path = harvested_data.sow_to_netcdf(output_dir=temp_output_dir)
        assert isinstance(path, str)

    def test_sow_with_transformations_keep_variables(self, temp_output_dir, harvested_data):
        """Test that variable selection transformations can be applied before sowing."""
        transformations = {"keep_variables": ["precip"]}
        transformed_data = apply_gridded_transformations(harvested_data, transformations)
        
        path = transformed_data.sow_to_netcdf(output_dir=temp_output_dir)
        
        assert Path(path).exists()
        
        # Verify transformation (temp should not exist)
        try:
            ds = xr.open_dataset(path, engine="h5netcdf")
            assert "precip" in ds.data_vars
            assert "temp" not in ds.data_vars
        finally:
            ds.close()