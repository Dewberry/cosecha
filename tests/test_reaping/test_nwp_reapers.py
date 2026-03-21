"""Tests for NWP (HRRR) reaper implementation."""

from __future__ import annotations

from datetime import datetime

import pytest

from cosecha.data_models import HarvestedData
from cosecha.reaping.exceptions import APIError, DateRangeError, ReaperError
from cosecha.reaping.nwp import HRRRReaper


@pytest.mark.requires_herbie
class TestHRRRReaper:
    """Test HRRRReaper implementation."""
    
    def test_initialization_valid(self):
        """Test valid initialization."""
        reaper = HRRRReaper(
            init_time="2026-01-01 00:00",
            forecast_hours=[1, 6, 12]
        )
        assert reaper.model == "hrrr"
        assert reaper.init_time == "2026-01-01 00:00"
        assert reaper.forecast_hours == [1, 6, 12]
    
    def test_initialization_with_range(self):
        """Test initialization with range for forecast_hours."""
        reaper = HRRRReaper(
            init_time="2026-01-01 00:00",
            forecast_hours=range(1, 13)
        )
        assert reaper.forecast_hours == list(range(1, 13))
    
    def test_initialization_custom_model(self):
        """Test initialization with custom model."""
        reaper = HRRRReaper(
            init_time="2026-01-01 00:00",
            forecast_hours=[1, 6],
            model="nam"
        )
        assert reaper.model == "nam"
    
    def test_invalid_init_time(self):
        """Test initialization fails with invalid init_time."""
        with pytest.raises(DateRangeError, match="Could not parse init_time"):
            HRRRReaper(
                init_time="invalid time",
                forecast_hours=[1, 6]
            )
    
    def test_invalid_forecast_hours_empty(self):
        """Test initialization fails with empty forecast_hours."""
        with pytest.raises(DateRangeError, match="cannot be empty"):
            HRRRReaper(
                init_time="2026-01-01 00:00",
                forecast_hours=[]
            )
    
    def test_invalid_forecast_hours_negative(self):
        """Test initialization fails with negative forecast_hours."""
        with pytest.raises(DateRangeError, match="must be positive integers"):
            HRRRReaper(
                init_time="2026-01-01 00:00",
                forecast_hours=[-1, 0, 1]
            )
    
    def test_invalid_forecast_hours_non_int(self):
        """Test initialization fails with non-integer forecast_hours."""
        with pytest.raises(DateRangeError, match="must be positive integers"):
            HRRRReaper(
                init_time="2026-01-01 00:00",
                forecast_hours=[1.5, 6, 12]
            )
    
    def test_validate_params_valid_dates(self):
        """Test _validate_params with valid dates."""
        reaper = HRRRReaper(
            init_time="2026-01-01 00:00",
            forecast_hours=[1, 6]
        )
        reaper._validate_params()  # Should not raise
    
    def test_validate_params_iso_format(self):
        """Test _validate_params with ISO format date."""
        reaper = HRRRReaper(
            init_time="2026-01-01T00:00:00Z",
            forecast_hours=[1, 6]
        )
        reaper._validate_params()  # Should not raise
    
    def test_reap_requires_herbie(self):
        """Test that reap raises APIError if herbie not available."""
        reaper = HRRRReaper(
            init_time="2026-01-01 00:00",
            forecast_hours=[1, 6]
        )
        
        # Simulate herbie not available
        reaper._herbie_available = False
        
        with pytest.raises(APIError, match="herbie is not installed"):
            reaper.reap()
    
    def test_reap_mocked_success(self, mocker):
        """Test reap with mocked FastHerbie."""
        pytest.importorskip("herbie")
        import xarray as xr
        
        # Create mock xarray Dataset
        mock_ds = xr.Dataset({
            "precip": (("time", "y", "x"), [[[1, 2], [3, 4]]]),
            "temp": (("time", "y", "x"), [[[20, 21], [22, 23]]]),
        })
        
        # Mock FastHerbie from herbie import
        mock_herbie = mocker.MagicMock()
        mock_herbie.return_value.xarray.return_value = mock_ds
        mocker.patch("herbie.FastHerbie", mock_herbie)
        
        reaper = HRRRReaper(
            init_time="2026-01-01 00:00",
            forecast_hours=[1, 6]
        )
        
        # Ensure herbie is marked as available
        reaper._herbie_available = True
        
        harvested = reaper.reap()
        
        assert isinstance(harvested, HarvestedData)
        assert harvested.source_name == "HRRR Forecast"
        assert harvested.is_gridded()
        assert not harvested.is_timeseries()
        assert len(harvested.variable_names) == 2
        assert "precip" in harvested.variable_names
        assert "temp" in harvested.variable_names
    
    def test_reap_returns_harvested_data(self, mocker):
        """Test that reap returns properly formatted HarvestedData."""
        pytest.importorskip("herbie")
        import xarray as xr
        
        mock_ds = xr.Dataset({
            "var1": (("y", "x"), [[1, 2], [3, 4]]),
        })
        
        mock_herbie = mocker.MagicMock()
        mock_herbie.return_value.xarray.return_value = mock_ds
        mocker.patch("herbie.FastHerbie", mock_herbie)
        
        reaper = HRRRReaper(
            init_time="2026-01-01 00:00",
            forecast_hours=[1, 6, 12]
        )
        reaper._herbie_available = True
        
        harvested = reaper.reap()
        
        # Check metadata
        assert harvested.metadata["model"] == "hrrr"
        assert harvested.metadata["init_time"] == "2026-01-01 00:00"
        assert harvested.metadata["forecast_hours"] == [1, 6, 12]
        assert "nwp_grid_points" in harvested.metadata
    
    def test_reap_api_error_handling(self, mocker):
        """Test that reap handles API errors gracefully."""
        pytest.importorskip("herbie")
        mock_herbie = mocker.MagicMock()
        mock_herbie.return_value.xarray.side_effect = Exception("Network error")
        mocker.patch("herbie.FastHerbie", mock_herbie)
        
        reaper = HRRRReaper(
            init_time="2026-01-01 00:00",
            forecast_hours=[1, 6]
        )
        reaper._herbie_available = True
        
        with pytest.raises(APIError, match="HRRR fetch failed"):
            reaper.reap()


@pytest.mark.requires_herbie
class TestGetVariableNames:
    """Test variable extraction from HRRR data."""
    
    def test_variable_names_extracted(self, mocker):
        """Test that variable names are correctly extracted from Dataset."""
        pytest.importorskip("herbie")
        import xarray as xr
        
        mock_ds = xr.Dataset({
            "precip": (("y", "x"), [[1, 2]]),
            "temp": (("y", "x"), [[20, 21]]),
            "wind_u": (("y", "x"), [[5, 6]]),
        })
        
        mock_herbie = mocker.MagicMock()
        mock_herbie.return_value.xarray.return_value = mock_ds
        mocker.patch("herbie.FastHerbie", mock_herbie)
        
        reaper = HRRRReaper(
            init_time="2026-01-01 00:00",
            forecast_hours=[1]
        )
        reaper._herbie_available = True
        
        harvested = reaper.reap()
        
        assert len(harvested.variable_names) == 3
        assert set(harvested.variable_names) == {"precip", "temp", "wind_u"}
    
    def test_empty_variables_handled(self, mocker):
        """Test handling of Dataset with no variables."""
        pytest.importorskip("herbie")
        import xarray as xr
        
        mock_ds = xr.Dataset()  # Empty dataset
        
        mock_herbie = mocker.MagicMock()
        mock_herbie.return_value.xarray.return_value = mock_ds
        mocker.patch("herbie.FastHerbie", mock_herbie)
        
        reaper = HRRRReaper(
            init_time="2026-01-01 00:00",
            forecast_hours=[1]
        )
        reaper._herbie_available = True
        
        # Empty datasets should raise an error during validation
        with pytest.raises(ReaperError, match="HRRR reaping failed"):
            reaper.reap()
