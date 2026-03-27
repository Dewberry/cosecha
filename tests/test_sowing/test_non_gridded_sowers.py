"""Tests for non-gridded sower implementations (Parquet, Iceberg)."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pytest
import xarray as xr

from cosecha.data_models import HarvestedData
from cosecha.reaping.utils import apply_ts_transformations
from cosecha.sowing.exceptions import SowerError
from cosecha.sowing.iceberg import IcebergSower
from cosecha.sowing.parquet import ParquetSower


class TestParquetSower:
    """Test ParquetSower implementation."""

    @pytest.fixture
    def temp_output_dir(self):
        """Temporary directory for output."""
        with TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_dataframe(self):
        """Sample DataFrame with time-series data."""
        return pd.DataFrame(
            {
                "time": pd.date_range("2026-01-01", periods=10, freq="D"),
                "site_id": ["01650000"] * 10,
                "value": [100.0, 105.5, 98.2, 112.3, 87.4, 95.1, 104.8, 110.2, 99.5, 102.1],
            }
        )

    @pytest.fixture
    def harvested_data(self, sample_dataframe):
        """Create HarvestedData with DataFrame."""
        return HarvestedData(
            data=sample_dataframe,
            source_name="USGS Streamflow",
            timestamp=pd.Timestamp("2026-01-11 12:00:00"),
            variable_names=["discharge_cfs"],
            metadata={"site_ids": ["01650000"], "parameter_code": "00060"},
        )

    def test_initialization(self, temp_output_dir):
        """Test ParquetSower initialization."""
        sower = ParquetSower(output_dir=temp_output_dir)
        assert sower.output_dir.exists()
        assert sower.output_dir.is_dir()

    def test_initialization_creates_directory(self, temp_output_dir):
        """Test that initialization creates directory if needed."""
        new_dir = temp_output_dir / "new" / "nested" / "dir"
        ParquetSower(output_dir=new_dir)
        assert new_dir.exists()

    def test_initialization_fails_with_file(self, temp_output_dir):
        """Test initialization fails if output_dir is a file."""
        file_path = temp_output_dir / "file.txt"
        file_path.touch()

        with pytest.raises(ValueError, match="must be a directory"):
            ParquetSower(output_dir=file_path)

    def test_validate_input_dataframe(self, temp_output_dir, harvested_data):
        """Test validation passes for DataFrame."""
        sower = ParquetSower(output_dir=temp_output_dir)
        sower._validate_input(harvested_data)  # Should not raise

    def test_validate_input_gridded_fails(self, temp_output_dir):
        """Test validation fails for gridded Dataset data."""
        sower = ParquetSower(output_dir=temp_output_dir)
        ds = xr.Dataset(
            {
                "precip": (("x", "y"), [[1, 2], [3, 4]]),
            }
        )

        data = HarvestedData(
            data=ds,
            source_name="HRRR Forecast",
            timestamp=pd.Timestamp("2026-01-01"),
            variable_names=["precip"],
            metadata={"init_time": "2026-01-01"},
        )

        with pytest.raises(SowerError, match="only supports time-series"):
            sower._validate_input(data)

    def test_sow_creates_parquet_file(self, temp_output_dir, harvested_data):
        """Test that sow creates a Parquet file."""
        path = harvested_data.sow_to_parquet(output_dir=temp_output_dir)

        assert Path(path).exists()
        assert path.endswith(".parquet")

    def test_sow_returns_path_string(self, temp_output_dir, harvested_data):
        """Test that sow returns a string path."""
        path = harvested_data.sow_to_parquet(output_dir=temp_output_dir)
        assert isinstance(path, str)

    def test_sow_with_transformations(self, temp_output_dir, harvested_data):
        """Test that transformations can be applied manually before sowing."""
        transformations = {"unit_conversions": {"value": 2.0}}
        transformed_data = apply_ts_transformations(harvested_data, transformations)
        
        path = transformed_data.sow_to_parquet(output_dir=temp_output_dir)
        
        assert Path(path).exists()
        
        # Verify transformation (value should be multiplied by 2.0)
        df_result = pd.read_parquet(path)
        assert df_result["value"].iloc[0] == 100.0 * 2.0  # original was 100.0


@pytest.mark.requires_iceberg
class TestIcebergSower:
    """Test IcebergSower implementation."""

    @pytest.fixture
    def temp_warehouse_dir(self):
        """Temporary directory for Iceberg warehouse."""
        with TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_dataframe(self):
        """Sample DataFrame with time-series data."""
        return pd.DataFrame(
            {
                "time": pd.date_range("2026-01-01", periods=10, freq="D"),
                "site_id": ["01650000"] * 10,
                "value": [100.0, 105.5, 98.2, 112.3, 87.4, 95.1, 104.8, 110.2, 99.5, 102.1],
            }
        )

    @pytest.fixture
    def harvested_data(self, sample_dataframe):
        """Create HarvestedData with DataFrame."""
        return HarvestedData(
            data=sample_dataframe,
            source_name="USGS Stagdata",
            timestamp=pd.Timestamp("2026-01-11 12:00:00"),
            variable_names=["gage_height_ft"],
            metadata={"site_ids": ["01650000"], "parameter_code": "00065"},
        )

    def test_initialization(self, temp_warehouse_dir):
        """Test IcebergSower initialization."""
        sower = IcebergSower(warehouse_path=temp_warehouse_dir)
        assert sower.warehouse_path.exists()
        assert sower.warehouse_path.is_dir()

    def test_initialization_creates_directory(self, temp_warehouse_dir):
        """Test that initialization creates directory if needed."""
        new_dir = temp_warehouse_dir / "ice" / "warehouse"
        IcebergSower(warehouse_path=new_dir)
        assert new_dir.exists()

    def test_initialization_fails_with_file(self, temp_warehouse_dir):
        """Test initialization fails if warehouse_path is a file."""
        file_path = temp_warehouse_dir / "file.txt"
        file_path.touch()

        with pytest.raises(ValueError, match="must be a directory"):
            IcebergSower(warehouse_path=file_path)

    def test_validate_input_dataframe(self, temp_warehouse_dir, harvested_data):
        """Test validation passes for DataFrame."""
        sower = IcebergSower(warehouse_path=temp_warehouse_dir)
        sower._validate_input(harvested_data)  # Should not raise

    def test_validate_input_gridded_fails(self, temp_warehouse_dir):
        """Test validation fails for gridded Dataset data."""
        sower = IcebergSower(warehouse_path=temp_warehouse_dir)
        ds = xr.Dataset(
            {
                "precip": (("x", "y"), [[1, 2], [3, 4]]),
            }
        )

        data = HarvestedData(
            data=ds,
            source_name="HRRR Forecast",
            timestamp=pd.Timestamp("2026-01-01"),
            variable_names=["precip"],
            metadata={"init_time": "2026-01-01"},
        )

        with pytest.raises(SowerError, match="only supports time-series"):
            sower._validate_input(data)
