"""Comprehensive tests for HarvestPipeline orchestrator.

Tests cover:
- Pipeline initialization
- Adding and validating reapers and sowers
- Complete harvest execution with and without transformations
- Error handling and propagation
- Multiple reaper/sower configuration
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest
import xarray as xr

from cosecha import (
    HarvestedData,
    HarvestPipeline,
)
from cosecha.data_models import HarvestedData
from cosecha.reaping.exceptions import APIError, ReaperError
from cosecha.sowing.exceptions import SowerError, WriteError


class TestHarvestPipelineInitialization:
    """Test HarvestPipeline initialization and configuration."""

    def test_initialization_empty(self) -> None:
        """Test that newly initialized pipeline is empty."""
        pipeline = HarvestPipeline()
        assert pipeline.reapers == []
        assert pipeline.sowers == []

    def test_add_reaper_valid(self) -> None:
        """Test adding a valid reaper to the pipeline."""
        pipeline = HarvestPipeline()
        reaper = MagicMock()
        reaper.reap = MagicMock(
            return_value=HarvestedData(
                data=pd.DataFrame({"a": [1, 2, 3]}),
                source_name="test",
                timestamp=datetime.now(UTC),
                variable_names=["a"],
            )
        )

        pipeline.add_reaper(reaper)
        assert len(pipeline.reapers) == 1
        assert pipeline.reapers[0] == reaper

    def test_add_reaper_invalid_no_reap_method(self) -> None:
        """Test that adding invalid reaper (no reap method) raises TypeError."""
        pipeline = HarvestPipeline()
        invalid_reaper = MagicMock(spec=[])  # No reap method

        with pytest.raises(TypeError, match="must implement.*protocol"):
            pipeline.add_reaper(invalid_reaper)

    def test_add_multiple_reapers(self) -> None:
        """Test adding multiple reapers to the pipeline."""
        pipeline = HarvestPipeline()
        reaper1 = MagicMock()
        reaper1.reap = MagicMock()
        reaper2 = MagicMock()
        reaper2.reap = MagicMock()

        pipeline.add_reaper(reaper1)
        pipeline.add_reaper(reaper2)
        assert len(pipeline.reapers) == 2

    def test_add_sower_valid(self) -> None:
        """Test adding a valid sower to the pipeline."""
        pipeline = HarvestPipeline()
        sower = MagicMock()
        sower.sow = MagicMock(return_value="./data/output.parquet")

        pipeline.add_sower(sower)
        assert len(pipeline.sowers) == 1
        assert pipeline.sowers[0] == sower

    def test_add_sower_invalid_no_sow_method(self) -> None:
        """Test that adding invalid sower (no sow method) raises TypeError."""
        pipeline = HarvestPipeline()
        invalid_sower = MagicMock(spec=[])  # No sow method

        with pytest.raises(TypeError, match="must implement.*protocol"):
            pipeline.add_sower(invalid_sower)

    def test_add_multiple_sowers(self) -> None:
        """Test adding multiple sowers to the pipeline."""
        pipeline = HarvestPipeline()
        sower1 = MagicMock()
        sower1.sow = MagicMock(return_value="./parquet.parquet")
        sower2 = MagicMock()
        sower2.sow = MagicMock(return_value="./zarr.zarr")

        pipeline.add_sower(sower1)
        pipeline.add_sower(sower2)
        assert len(pipeline.sowers) == 2


class TestHarvestPipelineExecution:
    """Test HarvestPipeline execution."""

    def test_execute_missing_reapers(self) -> None:
        """Test execute() raises ValueError if no reapers configured."""
        pipeline = HarvestPipeline()
        sower = MagicMock()
        sower.sow = MagicMock()
        pipeline.add_sower(sower)

        with pytest.raises(ValueError, match="no reapers configured"):
            pipeline.execute()

    def test_execute_missing_sowers(self) -> None:
        """Test execute() raises ValueError if no sowers configured."""
        pipeline = HarvestPipeline()
        reaper = MagicMock()
        reaper.reap = MagicMock()
        pipeline.add_reaper(reaper)

        with pytest.raises(ValueError, match="no sowers configured"):
            pipeline.execute()

    def test_execute_single_reaper_single_sower_timeseries(self) -> None:
        """Test execute with single reaper and sower for time-series data."""
        # Setup reaper to return time-series data
        df = pd.DataFrame(
            {
                "time": pd.date_range("2026-01-01", periods=3),
                "site_id": ["01018035", "01018035", "01018035"],
                "value": [1.5, 1.6, 1.7],
            }
        )
        harvested = HarvestedData(
            data=df,
            source_name="USGS_NWIS",
            timestamp=datetime.now(UTC),
            variable_names=["value"],
        )

        reaper = MagicMock()
        reaper.reap = MagicMock(return_value=harvested)

        # Setup sower to return a path
        sower = MagicMock()
        sower.sow = MagicMock(return_value="./data/nwis.parquet")

        # Execute pipeline
        pipeline = HarvestPipeline()
        pipeline.add_reaper(reaper)
        pipeline.add_sower(sower)
        paths = pipeline.execute()

        # Verify outputs
        assert len(paths) == 1
        assert paths[0] == "./data/nwis.parquet"
        reaper.reap.assert_called_once()
        sower.sow.assert_called_once_with(harvested, transformations=None)

    def test_execute_single_reaper_single_sower_gridded(self) -> None:
        """Test execute with single reaper and sower for gridded data."""
        # Setup reaper to return gridded data
        ds = xr.Dataset(
            {
                "tp": (["time", "lat", "lon"], [[[1, 2], [3, 4]]]),
            }
        )
        harvested = HarvestedData(
            data=ds,
            source_name="HRRR",
            timestamp=datetime.now(UTC),
            variable_names=["tp"],
        )

        reaper = MagicMock()
        reaper.reap = MagicMock(return_value=harvested)

        # Setup sower
        sower = MagicMock()
        sower.sow = MagicMock(return_value="./data/hrrr.zarr")

        # Execute pipeline
        pipeline = HarvestPipeline()
        pipeline.add_reaper(reaper)
        pipeline.add_sower(sower)
        paths = pipeline.execute()

        # Verify outputs
        assert len(paths) == 1
        assert paths[0] == "./data/hrrr.zarr"

    def test_execute_single_reaper_multiple_sowers(self) -> None:
        """Test execute with single reaper and multiple sowers."""
        # Setup reaper
        df = pd.DataFrame({"value": [1, 2, 3]})
        harvested = HarvestedData(
            data=df,
            source_name="test",
            timestamp=datetime.now(UTC),
            variable_names=["value"],
        )

        reaper = MagicMock()
        reaper.reap = MagicMock(return_value=harvested)

        # Setup multiple sowers
        sower1 = MagicMock()
        sower1.sow = MagicMock(return_value="./parquet.parquet")
        sower2 = MagicMock()
        sower2.sow = MagicMock(return_value="./iceberg.iceberg")

        # Execute pipeline
        pipeline = HarvestPipeline()
        pipeline.add_reaper(reaper)
        pipeline.add_sower(sower1)
        pipeline.add_sower(sower2)
        paths = pipeline.execute()

        # Verify that data was passed to both sowers
        assert len(paths) == 2
        assert "./parquet.parquet" in paths
        assert "./iceberg.iceberg" in paths
        sower1.sow.assert_called_once()
        sower2.sow.assert_called_once()

    def test_execute_with_transformations(self) -> None:
        """Test execute passes transformations to sower."""
        # Setup reaper
        df = pd.DataFrame({"flow": [1.0, 2.0, 3.0]})
        harvested = HarvestedData(
            data=df,
            source_name="test",
            timestamp=datetime.now(UTC),
            variable_names=["flow"],
        )

        reaper = MagicMock()
        reaper.reap = MagicMock(return_value=harvested)

        # Setup sower
        sower = MagicMock()
        sower.sow = MagicMock(return_value="./data.parquet")

        # Define transformations
        transformations = {"unit_conversions": {"flow": 0.0283168}}

        # Execute pipeline with transformations
        pipeline = HarvestPipeline()
        pipeline.add_reaper(reaper)
        pipeline.add_sower(sower)
        paths = pipeline.execute(transformations=transformations)

        # Verify transformations were passed to sower
        sower.sow.assert_called_once_with(harvested, transformations=transformations)
        assert len(paths) == 1


class TestHarvestPipelineErrorHandling:
    """Test error handling in HarvestPipeline."""

    def test_execute_reaper_error(self) -> None:
        """Test that reaper errors are propagated as ReaperError."""
        reaper = MagicMock()
        reaper.reap = MagicMock(side_effect=APIError("Connection timeout"))

        sower = MagicMock()
        sower.sow = MagicMock()

        pipeline = HarvestPipeline()
        pipeline.add_reaper(reaper)
        pipeline.add_sower(sower)

        with pytest.raises(APIError, match="Connection timeout"):
            pipeline.execute()

    def test_execute_reaper_unexpected_error(self) -> None:
        """Test that unexpected reaper errors are wrapped in ReaperError."""
        reaper = MagicMock()
        reaper.reap = MagicMock(side_effect=ValueError("Unexpected error in reaper"))

        sower = MagicMock()
        sower.sow = MagicMock()

        pipeline = HarvestPipeline()
        pipeline.add_reaper(reaper)
        pipeline.add_sower(sower)

        with pytest.raises(ReaperError, match="Unexpected error"):
            pipeline.execute()

    def test_execute_sower_error(self) -> None:
        """Test that sower errors are propagated as SowerError."""
        # Setup reaper
        df = pd.DataFrame({"value": [1, 2, 3]})
        harvested = HarvestedData(
            data=df,
            source_name="test",
            timestamp=datetime.now(UTC),
            variable_names=["value"],
        )

        reaper = MagicMock()
        reaper.reap = MagicMock(return_value=harvested)

        # Setup sower that fails
        sower = MagicMock()
        sower.sow = MagicMock(side_effect=WriteError("Disk full"))

        pipeline = HarvestPipeline()
        pipeline.add_reaper(reaper)
        pipeline.add_sower(sower)

        with pytest.raises(WriteError, match="Disk full"):
            pipeline.execute()

    def test_execute_sower_unexpected_error(self) -> None:
        """Test that unexpected sower errors are wrapped in SowerError."""
        # Setup reaper
        df = pd.DataFrame({"value": [1, 2, 3]})
        harvested = HarvestedData(
            data=df,
            source_name="test",
            timestamp=datetime.now(UTC),
            variable_names=["value"],
        )

        reaper = MagicMock()
        reaper.reap = MagicMock(return_value=harvested)

        # Setup sower that fails unexpectedly
        sower = MagicMock()
        sower.sow = MagicMock(side_effect=RuntimeError("Unexpected sower error"))

        pipeline = HarvestPipeline()
        pipeline.add_reaper(reaper)
        pipeline.add_sower(sower)

        with pytest.raises(SowerError, match="unexpected error"):
            pipeline.execute()

    def test_execute_first_sower_error_prevents_second_sower(self) -> None:
        """Test that error in first sower prevents second sower execution."""
        # Setup reaper
        df = pd.DataFrame({"value": [1, 2, 3]})
        harvested = HarvestedData(
            data=df,
            source_name="test",
            timestamp=datetime.now(UTC),
            variable_names=["value"],
        )

        reaper = MagicMock()
        reaper.reap = MagicMock(return_value=harvested)

        # Setup first sower that fails, second sower (not called)
        sower1 = MagicMock()
        sower1.sow = MagicMock(side_effect=SowerError("Sower 1 failed"))
        sower2 = MagicMock()
        sower2.sow = MagicMock(return_value="./data2.parquet")

        pipeline = HarvestPipeline()
        pipeline.add_reaper(reaper)
        pipeline.add_sower(sower1)
        pipeline.add_sower(sower2)

        with pytest.raises(SowerError):
            pipeline.execute()

        # Verify second sower was not called
        sower2.sow.assert_not_called()


class TestHarvestPipelineMultipleReapers:
    """Test HarvestPipeline with multiple reapers."""

    def test_execute_multiple_reapers_same_sower(self) -> None:
        """Test execute with multiple reapers writes each to same sower."""
        # Setup first reaper
        df1 = pd.DataFrame({"value": [1, 2, 3]})
        harvested1 = HarvestedData(
            data=df1,
            source_name="source1",
            timestamp=datetime.now(UTC),
            variable_names=["value"],
        )

        reaper1 = MagicMock()
        reaper1.reap = MagicMock(return_value=harvested1)

        # Setup second reaper
        df2 = pd.DataFrame({"value": [4, 5, 6]})
        harvested2 = HarvestedData(
            data=df2,
            source_name="source2",
            timestamp=datetime.now(UTC),
            variable_names=["value"],
        )

        reaper2 = MagicMock()
        reaper2.reap = MagicMock(return_value=harvested2)

        # Setup sower
        sower = MagicMock()
        sower.sow = MagicMock(
            side_effect=[
                "./data1.parquet",
                "./data2.parquet",
            ]
        )

        # Execute pipeline
        pipeline = HarvestPipeline()
        pipeline.add_reaper(reaper1)
        pipeline.add_reaper(reaper2)
        pipeline.add_sower(sower)
        paths = pipeline.execute()

        # Verify both reapers were called and both outputs were written
        assert len(paths) == 2
        assert sower.sow.call_count == 2
        reaper1.reap.assert_called_once()
        reaper2.reap.assert_called_once()


class TestHarvestPipelineLogging:
    """Test logging output from HarvestPipeline."""

    def test_execute_logs_pipeline_start(self, caplog) -> None:
        """Test that execute logs pipeline start."""
        with caplog.at_level(logging.INFO):
            # Setup reaper and sower
            df = pd.DataFrame({"value": [1, 2, 3]})
            harvested = HarvestedData(
                data=df,
                source_name="test",
                timestamp=datetime.now(UTC),
                variable_names=["value"],
            )

            reaper = MagicMock()
            reaper.reap = MagicMock(return_value=harvested)

            sower = MagicMock()
            sower.sow = MagicMock(return_value="./data.parquet")

            pipeline = HarvestPipeline()
            pipeline.add_reaper(reaper)
            pipeline.add_sower(sower)
            pipeline.execute()

        assert "Starting harvest pipeline" in caplog.text
        assert "1 reaper(s)" in caplog.text
        assert "1 sower(s)" in caplog.text

    def test_execute_logs_reaper_success(self, caplog) -> None:
        """Test that execute logs successful reaping."""
        with caplog.at_level(logging.INFO):
            df = pd.DataFrame({"value": [1, 2, 3]})
            harvested = HarvestedData(
                data=df,
                source_name="TEST_SOURCE",
                timestamp=datetime.now(UTC),
                variable_names=["value"],
            )

            reaper = MagicMock()
            reaper.reap = MagicMock(return_value=harvested)

            sower = MagicMock()
            sower.sow = MagicMock(return_value="./data.parquet")

            pipeline = HarvestPipeline()
            pipeline.add_reaper(reaper)
            pipeline.add_sower(sower)
            pipeline.execute()

        assert "Successfully reaped data from TEST_SOURCE" in caplog.text
        assert "1 variables" in caplog.text

    def test_execute_logs_sower_success(self, caplog) -> None:
        """Test that execute logs successful sowing."""
        with caplog.at_level(logging.INFO):
            df = pd.DataFrame({"value": [1, 2, 3]})
            harvested = HarvestedData(
                data=df,
                source_name="test",
                timestamp=datetime.now(UTC),
                variable_names=["value"],
            )

            reaper = MagicMock()
            reaper.reap = MagicMock(return_value=harvested)

            sower = MagicMock()
            sower.sow = MagicMock(return_value="./data/output.parquet")

            pipeline = HarvestPipeline()
            pipeline.add_reaper(reaper)
            pipeline.add_sower(sower)
            pipeline.execute()

        assert "Successfully sowed data" in caplog.text
        assert "./data/output.parquet" in caplog.text

    def test_execute_logs_pipeline_completion(self, caplog) -> None:
        """Test that execute logs pipeline completion."""
        with caplog.at_level(logging.INFO):
            df = pd.DataFrame({"value": [1, 2, 3]})
            harvested = HarvestedData(
                data=df,
                source_name="test",
                timestamp=datetime.now(UTC),
                variable_names=["value"],
            )

            reaper = MagicMock()
            reaper.reap = MagicMock(return_value=harvested)

            sower = MagicMock()
            sower.sow = MagicMock(return_value="./data.parquet")

            pipeline = HarvestPipeline()
            pipeline.add_reaper(reaper)
            pipeline.add_sower(sower)
            pipeline.execute()

        assert "completed successfully" in caplog.text
        assert "1 output(s)" in caplog.text
