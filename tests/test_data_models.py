"""Tests for data models."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest
import xarray as xr

from cosecha.data_models import (
    HarvestedData,
    validate_data,
    validate_date_range,
    validate_metadata,
    validate_spatial_bounds,
)


class TestValidateData:
    """Tests for validate_data function."""

    def test_validate_dataframe_valid(self):
        """Test validation of valid DataFrame."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        validate_data(df)  # Should not raise

    def test_validate_dataset_valid(self):
        """Test validation of valid xarray Dataset."""
        ds = xr.Dataset({"var": (["x", "y"], [[1, 2], [3, 4]])})
        validate_data(ds)  # Should not raise

    def test_validate_empty_dataframe(self):
        """Test validation fails for empty DataFrame."""
        df = pd.DataFrame()
        with pytest.raises(ValueError, match="DataFrame cannot be empty"):
            validate_data(df)

    def test_validate_empty_dataset(self):
        """Test validation fails for Dataset with no vars."""
        ds = xr.Dataset()
        with pytest.raises(ValueError, match="at least one data variable"):
            validate_data(ds)

    def test_validate_invalid_type(self):
        """Test validation fails for invalid data type."""
        with pytest.raises(TypeError, match=r"pd\.DataFrame or xr\.Dataset"):
            validate_data([1, 2, 3])


class TestValidateDateRange:
    """Tests for validate_date_range function."""

    def test_valid_date_range(self):
        """Test validation of valid date range."""
        validate_date_range("2026-01-01", "2026-01-31")  # Should not raise

    def test_same_dates(self):
        """Test validation with same start and end date."""
        validate_date_range("2026-01-01", "2026-01-01")  # Should not raise

    def test_invalid_start_date(self):
        """Test validation fails for invalid start date."""
        with pytest.raises(ValueError, match="Invalid date format"):
            validate_date_range("2026/01/01", "2026-01-31")

    def test_invalid_end_date(self):
        """Test validation fails for invalid end date."""
        with pytest.raises(ValueError, match="Invalid date format"):
            validate_date_range("2026-01-01", "2026/01/31")

    def test_start_after_end(self):
        """Test validation fails when start > end."""
        with pytest.raises(ValueError, match=r"start_date.*must be <= end_date"):
            validate_date_range("2026-01-31", "2026-01-01")


class TestValidateMetadata:
    """Tests for validate_metadata function."""

    def test_valid_metadata(self):
        """Test validation of valid metadata."""
        metadata = {
            "source_name": "TEST",
            "timestamp": datetime.now(UTC),
            "variable_names": ["var1"],
        }
        validate_metadata(metadata)  # Should not raise

    def test_missing_source_name(self):
        """Test validation fails when source_name missing."""
        metadata = {
            "timestamp": datetime.now(UTC),
            "variable_names": ["var1"],
        }
        with pytest.raises(ValueError, match="source_name"):
            validate_metadata(metadata)

    def test_missing_timestamp(self):
        """Test validation fails when timestamp missing."""
        metadata = {
            "source_name": "TEST",
            "variable_names": ["var1"],
        }
        with pytest.raises(ValueError, match="timestamp"):
            validate_metadata(metadata)

    def test_missing_variable_names(self):
        """Test validation fails when variable_names missing."""
        metadata = {
            "source_name": "TEST",
            "timestamp": datetime.now(UTC),
        }
        with pytest.raises(ValueError, match="variable_names"):
            validate_metadata(metadata)

    def test_invalid_variable_names_type(self):
        """Test validation fails when variable_names not a list."""
        metadata = {
            "source_name": "TEST",
            "timestamp": datetime.now(UTC),
            "variable_names": "var1",  # Should be list
        }
        with pytest.raises(TypeError, match="must be a list or tuple"):
            validate_metadata(metadata)


class TestValidateSpatialBounds:
    """Tests for validate_spatial_bounds function."""

    def test_valid_bounds(self):
        """Test validation of valid spatial bounds."""
        validate_spatial_bounds((-99.5, 31, -94.5, 34.5))  # Should not raise

    def test_valid_bounds_global(self):
        """Test validation of bounds covering larger area."""
        validate_spatial_bounds((-180, -90, 180, 90))  # Should not raise

    def test_longitude_out_of_range(self):
        """Test validation fails for out-of-range longitude."""
        with pytest.raises(ValueError, match="Longitude bounds"):
            validate_spatial_bounds((-200, 31, -94.5, 34.5))

    def test_latitude_out_of_range(self):
        """Test validation fails for out-of-range latitude."""
        with pytest.raises(ValueError, match="Latitude bounds"):
            validate_spatial_bounds((-99.5, -100, -94.5, 34.5))

    def test_inverted_longitude(self):
        """Test validation fails when lon_min > lon_max."""
        with pytest.raises(ValueError, match=r"lon_min.*lon_max"):
            validate_spatial_bounds((-94.5, 31, -99.5, 34.5))

    def test_inverted_latitude(self):
        """Test validation fails when lat_min > lat_max."""
        with pytest.raises(ValueError, match=r"lat_min.*lat_max"):
            validate_spatial_bounds((-99.5, 34.5, -94.5, 31))


class TestHarvestedData:
    """Tests for HarvestedData dataclass."""

    def test_create_with_dataframe(self):
        """Test creating HarvestedData with DataFrame."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        data = HarvestedData(
            data=df,
            source_name="TEST",
            timestamp=datetime.now(UTC),
            variable_names=["a", "b"],
        )
        assert data.is_timeseries()
        assert not data.is_gridded()
        assert data._data_type() == "timeseries"

    def test_create_with_dataset(self):
        """Test creating HarvestedData with xarray Dataset."""
        ds = xr.Dataset({"var": (["x", "y"], [[1, 2], [3, 4]])})
        data = HarvestedData(
            data=ds,
            source_name="TEST",
            timestamp=datetime.now(UTC),
            variable_names=["var"],
        )
        assert not data.is_timeseries()
        assert data.is_gridded()
        assert data._data_type() == "gridded"

    def test_invalid_data_type(self):
        """Test creating HarvestedData with invalid data type."""
        with pytest.raises(TypeError, match=r"pd\.DataFrame or xr\.Dataset"):
            HarvestedData(
                data=[1, 2, 3],  # Invalid
                source_name="TEST",
                timestamp=datetime.now(UTC),
                variable_names=["var"],
            )

    def test_empty_dataframe_fails(self):
        """Test creating HarvestedData with empty DataFrame fails."""
        with pytest.raises(ValueError, match="DataFrame cannot be empty"):
            HarvestedData(
                data=pd.DataFrame(),
                source_name="TEST",
                timestamp=datetime.now(UTC),
                variable_names=["var"],
            )

    def test_metadata_enrichment(self):
        """Test that metadata is enriched with required fields."""
        df = pd.DataFrame({"a": [1, 2]})
        ts = datetime.now(UTC)
        data = HarvestedData(
            data=df,
            source_name="TEST",
            timestamp=ts,
            variable_names=["a"],
            metadata={"extra": "value"},
        )
        assert data.metadata["source_name"] == "TEST"
        assert data.metadata["timestamp"] is ts
        assert data.metadata["variable_names"] == ["a"]
        assert data.metadata["extra"] == "value"

    def test_repr(self):
        """Test string representation."""
        df = pd.DataFrame({"a": [1, 2]})
        data = HarvestedData(
            data=df,
            source_name="TEST",
            timestamp=datetime.now(UTC),
            variable_names=["a"],
        )
        repr_str = repr(data)
        assert "TEST" in repr_str
        assert "timeseries" in repr_str
        assert "shape=" in repr_str
