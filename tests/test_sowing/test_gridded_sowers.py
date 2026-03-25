"""Tests for gridded sower implementations (Zarr, IceChunk, NetCDF)."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import xarray as xr

from cosecha.data_models import HarvestedData
from cosecha.sowing.icechunk import IceChunkSower
from cosecha.sowing.netcdf import NetCDFSower
from cosecha.sowing.zarr import ZarrSower
from cosecha.sowing.utils import apply_gridded_transformations

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
        return xr.Dataset({
            "precip": (("y", "x"), [[1.0, 2.0], [3.0, 4.0]]),
            "temp": (("y", "x"), [[20.0, 21.0], [19.0, 22.0]]),
        })
    
    @pytest.fixture
    def harvested_data(self, sample_dataset):
        """Create HarvestedData with xarray Dataset."""
        return HarvestedData(
            data=sample_dataset,
            source_name="HRRR Forecast",
            timestamp=__import__("pandas").Timestamp("2026-01-01 12:00:00"),
            variable_names=["precip", "temp"],
            metadata={"model": "HRRR", "init_time": "2026-01-01 00:00"}
        )
    
    def test_initialization(self, temp_output_dir):
        """Test ZarrSower initialization."""
        sower = ZarrSower(output_dir=temp_output_dir)
        assert sower.output_dir.exists()
        assert sower.output_dir.is_dir()
    
    def test_initialization_creates_directory(self, temp_output_dir):
        """Test that initialization creates directory if needed."""
        new_dir = temp_output_dir / "zarr" / "store"
        sower = ZarrSower(output_dir=new_dir)
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
        import pandas as pd
        
        sower = ZarrSower(output_dir=temp_output_dir)
        df = pd.DataFrame({
            "time": pd.date_range("2026-01-01", periods=5),
            "value": [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        
        data = HarvestedData(
            data=df,
            source_name="USGS Streamflow",
            timestamp=pd.Timestamp("2026-01-01"),
            variable_names=["discharge"],
            metadata={"site_id": "01650000"}
        )
        
        from cosecha.sowing.exceptions import SowerError
        with pytest.raises(SowerError, match="only supports gridded"):
            sower._validate_input(data)
    
    def test_apply_transformations_none(self, temp_output_dir, sample_dataset):
        """Test apply_transformations with None."""
        sower = ZarrSower(output_dir=temp_output_dir)
        result = apply_gridded_transformations(sample_dataset, transformations=None)
        xr.testing.assert_identical(result, sample_dataset)
    
    def test_apply_unit_conversions(self, temp_output_dir, sample_dataset):
        """Test unit conversion transformation."""
        sower = ZarrSower(output_dir=temp_output_dir)
        result = apply_gridded_transformations(
            sample_dataset,
            transformations={"unit_conversions": {"precip": 2.0}}
        )
        (result["precip"] == sample_dataset["precip"] * 2.0).all()
    
    def test_apply_variable_rename(self, temp_output_dir, sample_dataset):
        """Test variable rename transformation."""
        sower = ZarrSower(output_dir=temp_output_dir)
        result = apply_gridded_transformations(
            sample_dataset,
            transformations={"variable_rename": {"precip": "precipitation_mm"}}
        )
        assert "precipitation_mm" in result.data_vars
        assert "precip" not in result.data_vars
    
    def test_apply_keep_variables(self, temp_output_dir, sample_dataset):
        """Test keep variables transformation."""
        sower = ZarrSower(output_dir=temp_output_dir)
        result = apply_gridded_transformations(
            sample_dataset,
            transformations={"keep_variables": ["precip"]}
        )
        assert list(result.data_vars) == ["precip"]
    
    def test_sow_creates_zarr_store(self, temp_output_dir, harvested_data):
        """Test that sow creates a Zarr store."""
        sower = ZarrSower(output_dir=temp_output_dir)
        path = sower.sow(harvested_data)
        
        assert Path(path).exists()
        assert path.endswith(".zarr")
    
    def test_sow_returns_path_string(self, temp_output_dir, harvested_data):
        """Test that sow returns a string path."""
        sower = ZarrSower(output_dir=temp_output_dir)
        path = sower.sow(harvested_data)
        assert isinstance(path, str)
    
    def test_sow_with_transformations(self, temp_output_dir, harvested_data):
        """Test sow with transformations."""
        sower = ZarrSower(output_dir=temp_output_dir)
        path = sower.sow(
            harvested_data,
            transformations={"unit_conversions": {"precip": 0.5}}
        )
        assert Path(path).exists()


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
        return xr.Dataset({
            "precip": (("y", "x"), [[1.0, 2.0], [3.0, 4.0]]),
            "temp": (("y", "x"), [[20.0, 21.0], [19.0, 22.0]]),
        })
    
    @pytest.fixture
    def harvested_data(self, sample_dataset):
        """Create HarvestedData with xarray Dataset."""
        import pandas as pd
        return HarvestedData(
            data=sample_dataset,
            source_name="HRRR Forecast",
            timestamp=pd.Timestamp("2026-01-01 12:00:00"),
            variable_names=["precip", "temp"],
            metadata={"model": "HRRR", "init_time": "2026-01-01 00:00"}
        )
    
    def test_initialization(self, temp_storage_dir):
        """Test IceChunkSower initialization."""
        sower = IceChunkSower(storage_path=temp_storage_dir)
        assert sower.storage_path.exists()
        assert sower.storage_path.is_dir()
    
    def test_initialization_creates_directory(self, temp_storage_dir):
        """Test that initialization creates directory if needed."""
        new_dir = temp_storage_dir / "ice" / "chunk" / "store"
        sower = IceChunkSower(storage_path=new_dir)
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
        import pandas as pd
        
        sower = IceChunkSower(storage_path=temp_storage_dir)
        df = pd.DataFrame({
            "time": pd.date_range("2026-01-01", periods=5),
            "value": [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        
        data = HarvestedData(
            data=df,
            source_name="USGS Streamflow",
            timestamp=pd.Timestamp("2026-01-01"),
            variable_names=["discharge"],
            metadata={"site_id": "01650000"}
        )
        
        from cosecha.sowing.exceptions import SowerError
        with pytest.raises(SowerError, match="only supports gridded"):
            sower._validate_input(data)
    
    def test_apply_transformations_none(self, temp_storage_dir, sample_dataset):
        """Test apply_transformations with None."""
        sower = IceChunkSower(storage_path=temp_storage_dir)
        result = apply_gridded_transformations(sample_dataset, transformations=None)
        xr.testing.assert_identical(result, sample_dataset)
    
    def test_sow_creates_icechunk_store(self, temp_storage_dir, harvested_data):
        """Test that sow creates an IceChunk store."""
        sower = IceChunkSower(storage_path=temp_storage_dir)
        path = sower.sow(harvested_data)
        
        # IceChunk manages its own internal layout, so we check if the storage root exists
        # and has populated data rather than assuming a Zarr v2 style literal directory path
        assert sower.storage_path.exists()
        assert any(sower.storage_path.iterdir())
    
    def test_sow_returns_path_string(self, temp_storage_dir, harvested_data):
        """Test that sow returns a string path."""
        sower = IceChunkSower(storage_path=temp_storage_dir)
        path = sower.sow(harvested_data)
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
        return xr.Dataset({
            "precip": (("y", "x"), [[1.0, 2.0], [3.0, 4.0]]),
            "temp": (("y", "x"), [[20.0, 21.0], [19.0, 22.0]]),
        })
    
    @pytest.fixture
    def harvested_data(self, sample_dataset):
        """Create HarvestedData with xarray Dataset."""
        return HarvestedData(
            data=sample_dataset,
            source_name="HRRR Forecast",
            timestamp=__import__("pandas").Timestamp("2026-01-01 12:00:00"),
            variable_names=["precip", "temp"],
            metadata={"model": "HRRR", "init_time": "2026-01-01 00:00"}
        )
    
    def test_initialization(self, temp_output_dir):
        """Test NetCDFSower initialization."""
        sower = NetCDFSower(output_dir=temp_output_dir)
        assert sower.output_dir.exists()
        assert sower.output_dir.is_dir()
    
    def test_initialization_creates_directory(self, temp_output_dir):
        """Test that initialization creates directory if needed."""
        new_dir = temp_output_dir / "netcdf" / "output"
        sower = NetCDFSower(output_dir=new_dir)
        assert new_dir.exists()
    
    def test_initialization_with_compression(self, temp_output_dir):
        """Test initialization with custom compression settings."""
        sower = NetCDFSower(
            output_dir=temp_output_dir,
            compression="zlib",
            compression_level=7
        )
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
        import pandas as pd
        
        sower = NetCDFSower(output_dir=temp_output_dir)
        df = pd.DataFrame({
            "time": pd.date_range("2026-01-01", periods=5),
            "value": [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        
        data = HarvestedData(
            data=df,
            source_name="USGS Streamflow",
            timestamp=pd.Timestamp("2026-01-01"),
            variable_names=["discharge"],
            metadata={"site_id": "01650000"}
        )
        
        from cosecha.sowing.exceptions import SowerError
        with pytest.raises(SowerError, match="only supports gridded"):
            sower._validate_input(data)
    
    def test_apply_transformations_none(self, temp_output_dir, sample_dataset):
        """Test apply_transformations with None."""
        sower = NetCDFSower(output_dir=temp_output_dir)
        result = apply_gridded_transformations(sample_dataset, transformations=None)
        xr.testing.assert_identical(result, sample_dataset)
    
    def test_apply_unit_conversions(self, temp_output_dir, sample_dataset):
        """Test unit conversion transformation."""
        sower = NetCDFSower(output_dir=temp_output_dir)
        result = apply_gridded_transformations(
            sample_dataset,
            transformations={"unit_conversions": {"precip": 2.0}}
        )
        assert (result["precip"] == sample_dataset["precip"] * 2.0).all()
    
    def test_apply_variable_rename(self, temp_output_dir, sample_dataset):
        """Test variable rename transformation."""
        sower = NetCDFSower(output_dir=temp_output_dir)
        result = apply_gridded_transformations(
            sample_dataset,
            transformations={"variable_rename": {"precip": "precipitation_mm"}}
        )
        assert "precipitation_mm" in result.data_vars
        assert "precip" not in result.data_vars
    
    def test_apply_keep_variables(self, temp_output_dir, sample_dataset):
        """Test keep variables transformation."""
        sower = NetCDFSower(output_dir=temp_output_dir)
        result = apply_gridded_transformations(
            sample_dataset,
            transformations={"keep_variables": ["precip"]}
        )
        assert list(result.data_vars) == ["precip"]
    
    def test_sow_creates_netcdf_file(self, temp_output_dir, harvested_data):
        """Test that sow creates a NetCDF file."""
        sower = NetCDFSower(output_dir=temp_output_dir)
        path = sower.sow(harvested_data)
        
        assert Path(path).exists()
        assert path.endswith(".nc")
    
    def test_sow_returns_path_string(self, temp_output_dir, harvested_data):
        """Test that sow returns a string path."""
        sower = NetCDFSower(output_dir=temp_output_dir)
        path = sower.sow(harvested_data)
        assert isinstance(path, str)
    
    def test_sow_with_transformations(self, temp_output_dir, harvested_data):
        """Test sow with transformations."""
        sower = NetCDFSower(output_dir=temp_output_dir)
        path = sower.sow(
            harvested_data,
            transformations={"unit_conversions": {"precip": 0.5}}
        )
        assert Path(path).exists()
