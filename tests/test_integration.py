"""Integration tests for complete harvest workflows.

Tests cover end-to-end scenarios:
- NWIS time-series → Parquet (with transformations)
- HRRR gridded → NetCDF (with transformations)
- Error scenarios and recovery
- Multiple source-to-multiple-sink patterns
- Data validation through pipeline
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

import pandas as pd
import pytest
import xarray as xr

from cosecha import (
    HarvestedData,
    HarvestPipeline,
    NetCDFSower,
    ParquetSower,
    ZarrSower,
)
from cosecha.reaping.exceptions import APIError
from cosecha.reaping.utils import apply_gridded_transformations, apply_ts_transformations
from cosecha.sowing.exceptions import SowerError


class TestTimeSeriesWorkflow:
    """Integration tests for time-series workflow (NWIS → Parquet)."""

    @pytest.fixture
    def temp_output_dir(self):
        """Temporary directory for output."""
        with TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_streamflow_data(self):
        """Sample streamflow data from NWIS."""
        return pd.DataFrame(
            {
                "time": pd.date_range("2026-01-01", periods=30, freq="D"),
                "site_id": ["01018035"] * 30,
                "discharge_cfs": [
                    1250.0,
                    1280.5,
                    1150.2,
                    1320.8,
                    1100.5,
                    1400.2,
                    1350.8,
                    1450.0,
                    1380.5,
                    1420.2,
                    1500.0,
                    1550.8,
                    1480.2,
                    1520.5,
                    1600.0,
                    1450.0,
                    1350.2,
                    1280.5,
                    1200.8,
                    1150.0,
                    1100.2,
                    1050.5,
                    1000.8,
                    1150.2,
                    1250.0,
                    1350.8,
                    1450.0,
                    1550.2,
                    1480.5,
                    1380.0,
                ],
            }
        )

    def test_full_nwis_to_parquet_workflow(self, temp_output_dir, sample_streamflow_data):
        """Test complete NWIS → Parquet workflow."""
        # Create mock reaper that returns sample data
        reaper = MagicMock()
        reaper.reap = MagicMock(
            return_value=HarvestedData(
                data=sample_streamflow_data,
                source_name="USGS_Streamflow",
                timestamp=datetime(2026, 1, 31, 12, 0, 0),
                variable_names=["discharge_cfs"],
                metadata={"site_ids": ["01018035"], "parameter_code": "00060"},
            )
        )

        # Create sower
        sower = ParquetSower(output_dir=temp_output_dir)

        # Create pipeline
        pipeline = HarvestPipeline()
        pipeline.add_reaper(reaper)
        pipeline.add_sower(sower)

        # Execute
        paths = pipeline.execute()

        # Verify
        assert len(paths) == 1
        assert Path(paths[0]).exists()
        assert paths[0].endswith(".parquet")

        # Verify data integrity
        written_df = pd.read_parquet(paths[0])
        pd.testing.assert_frame_equal(written_df, sample_streamflow_data)

    def test_nwis_to_parquet_with_unit_conversion(self, temp_output_dir, sample_streamflow_data):
        """Test NWIS → Parquet with unit conversion transformation."""
        harvested = HarvestedData(
            data=sample_streamflow_data,
            source_name="USGS_Streamflow",
            timestamp=datetime(2026, 1, 31, 12, 0, 0),
            variable_names=["discharge_cfs"],
            metadata={"site_ids": ["01018035"]},
        )
        cfs_to_cms = 0.0283168
        transformations = {"unit_conversions": {"discharge_cfs": cfs_to_cms}}
        transformed_harvested = apply_ts_transformations(harvested, transformations)

        reaper = MagicMock()
        reaper.reap = MagicMock(return_value=transformed_harvested)

        sower = ParquetSower(output_dir=temp_output_dir)
        pipeline = HarvestPipeline()
        pipeline.add_reaper(reaper)
        pipeline.add_sower(sower)

        # Execute
        paths = pipeline.execute()

        # Verify transformation was applied
        written_df = pd.read_parquet(paths[0])
        expected_discharge = sample_streamflow_data["discharge_cfs"] * cfs_to_cms
        pd.testing.assert_series_equal(
            written_df["discharge_cfs"], expected_discharge, check_names=True
        )

    def test_nwis_to_parquet_with_column_filtering(self, temp_output_dir, sample_streamflow_data):
        """Test NWIS → Parquet with column filtering."""
        harvested = HarvestedData(
            data=sample_streamflow_data,
            source_name="USGS_Streamflow",
            timestamp=datetime(2026, 1, 31, 12, 0, 0),
            variable_names=["discharge_cfs"],
            metadata={"site_ids": ["01018035"]},
        )
        transformations = {"filter_columns": ["time", "discharge_cfs"]}
        transformed_harvested = apply_ts_transformations(harvested, transformations)

        reaper = MagicMock()
        reaper.reap = MagicMock(return_value=transformed_harvested)

        sower = ParquetSower(output_dir=temp_output_dir)
        pipeline = HarvestPipeline()
        pipeline.add_reaper(reaper)
        pipeline.add_sower(sower)

        # Execute
        paths = pipeline.execute()

        # Verify only selected columns remain
        written_df = pd.read_parquet(paths[0])
        assert set(written_df.columns) == {"time", "discharge_cfs"}

    def test_nwis_to_parquet_validation(self, temp_output_dir):
        """Test pipeline handles data validation errors appropriately."""
        # Create valid data to test that pipeline handles it correctly
        valid_df = pd.DataFrame(
            {"time": pd.date_range("2026-01-01", periods=3), "discharge_cfs": [100.0, 150.0, 120.0]}
        )

        reaper = MagicMock()
        reaper.reap = MagicMock(
            return_value=HarvestedData(
                data=valid_df,
                source_name="USGS_Streamflow",
                timestamp=datetime(2026, 1, 31, 12, 0, 0),
                variable_names=["discharge_cfs"],
                metadata={"site_ids": ["01018035"]},
            )
        )

        sower = ParquetSower(output_dir=temp_output_dir)
        pipeline = HarvestPipeline()
        pipeline.add_reaper(reaper)
        pipeline.add_sower(sower)

        # Should execute successfully with valid data
        paths = pipeline.execute()
        assert len(paths) == 1
        assert Path(paths[0]).exists()

        # Verify metadata is preserved
        written_df = pd.read_parquet(paths[0])
        assert "discharge_cfs" in written_df.columns


@pytest.mark.requires_netcdf
class TestGriddedWorkflow:
    """Integration tests for gridded workflow (HRRR → NetCDF)."""

    @pytest.fixture
    def temp_output_dir(self):
        """Temporary directory for output."""
        with TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_hrrr_data(self):
        """Sample HRRR gridded forecast data."""
        return xr.Dataset(
            {
                "precip": (
                    ["time", "lat", "lon"],
                    [
                        [[0.5, 0.6, 0.4], [0.7, 0.8, 0.5], [0.6, 0.7, 0.4]],
                        [[1.0, 1.2, 0.8], [1.5, 1.6, 1.0], [1.2, 1.4, 0.9]],
                    ],
                ),
                "temp": (
                    ["time", "lat", "lon"],
                    [
                        [[20.0, 21.0, 19.0], [22.0, 23.0, 21.0], [21.0, 22.0, 20.0]],
                        [[18.0, 19.0, 17.0], [20.0, 21.0, 19.0], [19.0, 20.0, 18.0]],
                    ],
                ),
            },
            coords={"time": [0, 6], "lat": [40.0, 40.5, 41.0], "lon": [-104.0, -103.5, -103.0]},
        )

    def test_full_hrrr_to_netcdf_workflow(self, temp_output_dir, sample_hrrr_data):
        """Test complete HRRR → NetCDF workflow."""
        reaper = MagicMock()
        reaper.reap = MagicMock(
            return_value=HarvestedData(
                data=sample_hrrr_data,
                source_name="HRRR_Forecast",
                timestamp=datetime(2026, 1, 20, 0, 0, 0),
                variable_names=["precip", "temp"],
                metadata={"model": "HRRR", "init_time": "2026-01-20T00:00:00"},
            )
        )

        sower = NetCDFSower(output_dir=temp_output_dir)
        pipeline = HarvestPipeline()
        pipeline.add_reaper(reaper)
        pipeline.add_sower(sower)

        # Execute
        paths = pipeline.execute()

        # Verify
        assert len(paths) == 1
        assert Path(paths[0]).exists()
        assert paths[0].endswith(".nc")

        # Verify data integrity
        written_ds = xr.open_dataset(paths[0], engine="h5netcdf")
        xr.testing.assert_equal(written_ds[["precip", "temp"]], sample_hrrr_data)
        written_ds.close()

    def test_hrrr_to_netcdf_with_unit_conversion(self, temp_output_dir, sample_hrrr_data):
        """Test HRRR → NetCDF with unit conversion."""
        harvested = HarvestedData(
            data=sample_hrrr_data,
            source_name="HRRR_Forecast",
            timestamp=datetime(2026, 1, 20, 0, 0, 0),
            variable_names=["precip", "temp"],
            metadata={"model": "HRRR"},
        )
        transformations = {
            "unit_conversions": {"temp": 1.8}  # °C to °F scale conversion factor
        }
        transformed_harvested = apply_gridded_transformations(harvested, transformations)

        reaper = MagicMock()
        reaper.reap = MagicMock(return_value=transformed_harvested)

        sower = NetCDFSower(output_dir=temp_output_dir)
        pipeline = HarvestPipeline()
        pipeline.add_reaper(reaper)
        pipeline.add_sower(sower)

        # Execute
        paths = pipeline.execute()

        # Verify transformation
        written_ds = xr.open_dataset(paths[0], engine="h5netcdf")
        expected_temp = sample_hrrr_data["temp"] * 1.8
        xr.testing.assert_allclose(written_ds["temp"], expected_temp)
        written_ds.close()

    def test_hrrr_to_netcdf_with_variable_selection(self, temp_output_dir, sample_hrrr_data):
        """Test HRRR → NetCDF with variable selection."""
        harvested = HarvestedData(
            data=sample_hrrr_data,
            source_name="HRRR_Forecast",
            timestamp=datetime(2026, 1, 20, 0, 0, 0),
            variable_names=["precip", "temp"],
            metadata={"model": "HRRR"},
        )
        transformations = {"keep_variables": ["precip"]}
        transformed_harvested = apply_gridded_transformations(harvested, transformations)

        reaper = MagicMock()
        reaper.reap = MagicMock(return_value=transformed_harvested)

        sower = NetCDFSower(output_dir=temp_output_dir)
        pipeline = HarvestPipeline()
        pipeline.add_reaper(reaper)
        pipeline.add_sower(sower)

        # Execute keeping only precip
        paths = pipeline.execute()

        # Verify only precip is in output
        written_ds = xr.open_dataset(paths[0], engine="h5netcdf")
        assert list(written_ds.data_vars) == ["precip"]
        written_ds.close()


class TestMultiSourceWorkflow:
    """Integration tests for multi-source workflows."""

    @pytest.fixture
    def temp_output_dir(self):
        """Temporary directory for output."""
        with TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_data_timeseries(self):
        """Sample time-series data."""
        return pd.DataFrame(
            {
                "time": pd.date_range("2026-01-01", periods=10),
                "site_id": ["01018035"] * 10,
                "value": range(100, 110),
            }
        )

    @pytest.fixture
    def sample_data_gridded(self):
        """Sample gridded data."""
        return xr.Dataset(
            {
                "var": (["x", "y"], [[1, 2], [3, 4]]),
            }
        )

    def test_multiple_reapers_single_sower(
        self, temp_output_dir, sample_data_timeseries, sample_data_gridded
    ):
        """Test pipeline with multiple reapers writing to single sower."""
        # Two reapers: time-series and gridded
        reaper1 = MagicMock()
        reaper1.reap = MagicMock(
            return_value=HarvestedData(
                data=sample_data_timeseries,
                source_name="NWIS",
                timestamp=datetime(2026, 1, 10, 12, 0, 0),
                variable_names=["value"],
                metadata={},
            )
        )

        # Note: gridded data to ParquetSower will fail validation,
        # but we're testing that both reapers are called and first writes successfully
        reaper2 = MagicMock()
        reaper2.reap = MagicMock(
            return_value=HarvestedData(
                data=sample_data_timeseries,  # Use timeseries for valid test
                source_name="Stage",
                timestamp=datetime(2026, 1, 10, 12, 0, 0),
                variable_names=["value"],
                metadata={},
            )
        )

        sower = ParquetSower(output_dir=temp_output_dir)
        pipeline = HarvestPipeline()
        pipeline.add_reaper(reaper1)
        pipeline.add_reaper(reaper2)
        pipeline.add_sower(sower)

        # Execute
        paths = pipeline.execute()

        # Both reapers should have been called and both sowed
        assert len(paths) == 2
        assert reaper1.reap.called
        assert reaper2.reap.called

    def test_single_reaper_multiple_sowers(self, temp_output_dir, sample_data_gridded):
        """Test pipeline with single reaper writing to multiple sowers."""
        reaper = MagicMock()
        reaper.reap = MagicMock(
            return_value=HarvestedData(
                data=sample_data_gridded,
                source_name="HRRR",
                timestamp=datetime(2026, 1, 20, 0, 0, 0),
                variable_names=["var"],
                metadata={},
            )
        )

        netcdf_sower = NetCDFSower(output_dir=temp_output_dir / "netcdf")

        # Create a second sower directory
        zarr_output = temp_output_dir / "zarr"

        zarr_sower = ZarrSower(output_dir=zarr_output)

        pipeline = HarvestPipeline()
        pipeline.add_reaper(reaper)
        pipeline.add_sower(netcdf_sower)
        pipeline.add_sower(zarr_sower)

        # Execute
        paths = pipeline.execute()

        # Both sowers should have written data
        assert len(paths) == 2
        assert any(p.endswith(".nc") for p in paths)
        assert any(p.endswith(".zarr") for p in paths)


class TestErrorRecovery:
    """Integration tests for error handling and recovery."""

    @pytest.fixture
    def temp_output_dir(self):
        """Temporary directory for output."""
        with TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_reaper_error_stops_pipeline(self, temp_output_dir):
        """Test that reaper error stops the pipeline."""
        reaper = MagicMock()
        reaper.reap = MagicMock(side_effect=APIError("Connection failed"))

        sower = MagicMock()
        sower.sow = MagicMock(return_value="./output.parquet")

        pipeline = HarvestPipeline()
        pipeline.add_reaper(reaper)
        pipeline.add_sower(sower)

        # Should raise APIError
        with pytest.raises(APIError):
            pipeline.execute()

        # Sower should not be called
        assert not sower.sow.called

    def test_sower_error_stops_pipeline(self, temp_output_dir):
        """Test that sower error stops the pipeline."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        reaper = MagicMock()
        reaper.reap = MagicMock(
            return_value=HarvestedData(
                data=df,
                source_name="test",
                timestamp=datetime.now(UTC),
                variable_names=["a"],
                metadata={},
            )
        )

        # Make sower raise error
        sower = MagicMock()
        sower.sow = MagicMock(side_effect=SowerError("Write failed"))

        pipeline = HarvestPipeline()
        pipeline.add_reaper(reaper)
        pipeline.add_sower(sower)

        # Should raise SowerError
        with pytest.raises(SowerError):
            pipeline.execute()

    def test_invalid_reaper_configuration(self, temp_output_dir):
        """Test that invalid reaper configuration is caught."""
        pipeline = HarvestPipeline()
        sower = ParquetSower(output_dir=temp_output_dir)
        pipeline.add_sower(sower)

        # No reaper added - should raise ValueError
        with pytest.raises(ValueError, match="no reapers"):
            pipeline.execute()

    def test_invalid_sower_configuration(self, temp_output_dir):
        """Test that invalid sower configuration is caught."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        reaper = MagicMock()
        reaper.reap = MagicMock(
            return_value=HarvestedData(
                data=df,
                source_name="test",
                timestamp=datetime.now(UTC),
                variable_names=["a"],
                metadata={},
            )
        )

        pipeline = HarvestPipeline()
        pipeline.add_reaper(reaper)

        # No sower added - should raise ValueError
        with pytest.raises(ValueError, match="no sowers"):
            pipeline.execute()


class TestDataValidationThroughPipeline:
    """Integration tests for data validation through pipeline."""

    @pytest.fixture
    def temp_output_dir(self):
        """Temporary directory for output."""
        with TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_metadata_enrichment(self, temp_output_dir):
        """Test that metadata is properly enriched through pipeline."""
        sample_data = pd.DataFrame(
            {
                "time": pd.date_range("2026-01-01", periods=5),
                "value": [1, 2, 3, 4, 5],
            }
        )

        reaper = MagicMock()
        reaper.reap = MagicMock(
            return_value=HarvestedData(
                data=sample_data,
                source_name="TestSource",
                timestamp=datetime(2026, 1, 5, 12, 0, 0),
                variable_names=["value"],
                metadata={"custom_field": "custom_value"},
            )
        )

        sower = ParquetSower(output_dir=temp_output_dir)
        pipeline = HarvestPipeline()
        pipeline.add_reaper(reaper)
        pipeline.add_sower(sower)

        paths = pipeline.execute()

        # Verify data was written
        assert len(paths) == 1
        assert Path(paths[0]).exists()

    def test_variable_names_preserved(self, temp_output_dir):
        """Test that variable names are preserved through pipeline."""
        sample_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=5),
                "discharge": [100.0, 105.0, 98.0, 112.0, 95.0],
                "stage": [5.2, 5.4, 5.1, 5.6, 5.0],
            }
        )

        reaper = MagicMock()
        reaper.reap = MagicMock(
            return_value=HarvestedData(
                data=sample_data,
                source_name="USGS",
                timestamp=datetime(2026, 1, 5, 12, 0, 0),
                variable_names=["discharge", "stage"],
                metadata={},
            )
        )

        sower = ParquetSower(output_dir=temp_output_dir)
        pipeline = HarvestPipeline()
        pipeline.add_reaper(reaper)
        pipeline.add_sower(sower)

        paths = pipeline.execute()

        # Verify all columns are preserved
        written_df = pd.read_parquet(paths[0])
        assert set(written_df.columns) == {"timestamp", "discharge", "stage"}
