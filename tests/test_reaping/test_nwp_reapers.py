"""Tests for NWP (HRRR) reaper implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from cosecha.exceptions import APIError, DateRangeError
from cosecha.reaping.nwp import NWPReaper


def _create_mock_hrrr_dataset(variables: list[str] | None = None):

    if variables is None:
        variables = ["tp"]

    data_vars = {
        var: (["step", "y", "x"], np.zeros((1, 2, 2), dtype=np.float32)) for var in variables
    }

    return xr.Dataset(
        data_vars=data_vars,
        coords={
            "step": (["step"], pd.to_timedelta([1], unit="h")),
            "valid_time": (["step"], [pd.Timestamp("2026-03-23T01:00")]),
            "latitude": (["y", "x"], [[21.14, 21.15], [21.16, 21.17]]),
            "longitude": (["y", "x"], [[-122.7, -122.6], [-122.5, -122.4]]),
            "time": pd.Timestamp("2026-03-23"),
            "surface": 0.0,
        },
        attrs={
            "model": "hrrr",
            "product": "sfc",
        },
    )


@pytest.mark.requires_herbie
class TestNWPReaper:
    """Test NWPReaper implementation."""

    def test_initialization_valid(self):
        """Test valid initialization."""
        reaper = NWPReaper(init_time="2026-01-01 00:00", forecast_hours=[1, 6, 12])
        assert reaper.model == "hrrr"
        assert reaper.init_time == "2026-01-01 00:00"
        assert reaper.forecast_hours == [1, 6, 12]

    def test_initialization_with_range(self):
        """Test initialization with range for forecast_hours."""
        reaper = NWPReaper(init_time="2026-01-01 00:00", forecast_hours=range(1, 13))
        assert reaper.forecast_hours == list(range(1, 13))

    def test_initialization_custom_model(self):
        """Test initialization with custom model."""
        reaper = NWPReaper(
            init_time="2026-01-01 00:00", forecast_hours=[1, 6], model="rrfs", search_str="custom"
        )
        assert reaper.model == "rrfs"

    def test_invalid_init_time(self):
        """Test initialization fails with invalid init_time."""
        with pytest.raises(DateRangeError, match="Could not parse init_time"):
            NWPReaper(init_time="invalid time", forecast_hours=[1, 6])

    def test_valid_forecast_hours_empty(self):
        """Test initialization with empty forecast_hours."""
        reaper = NWPReaper(init_time="2026-01-01 00:00", forecast_hours=[])
        assert reaper.forecast_hours == []

    def test_invalid_forecast_hours_negative(self):
        """Test initialization fails with negative forecast_hours."""
        with pytest.raises(DateRangeError, match="must be positive integers"):
            NWPReaper(init_time="2026-01-01 00:00", forecast_hours=[-1, 0, 1])

    def test_invalid_forecast_hours_non_int(self):
        """Test initialization fails with non-integer forecast_hours."""
        with pytest.raises(DateRangeError, match="must be positive integers"):
            NWPReaper(init_time="2026-01-01 00:00", forecast_hours=[1.5, 6, 12])

    def test_validate_params_valid_dates(self):
        """Test _validate_params with valid dates."""
        reaper = NWPReaper(init_time="2026-01-01 00:00", forecast_hours=[1, 6])
        reaper._validate_params()  # Should not raise

    def test_validate_params_iso_format(self):
        """Test _validate_params with ISO format date."""
        reaper = NWPReaper(init_time="2026-01-01T00:00:00Z", forecast_hours=[1, 6])
        reaper._validate_params()  # Should not raise

    def test_reap_requires_herbie(self, mocker):
        """Test that reap raises APIError if herbie not available."""
        reaper = NWPReaper(init_time="2026-01-01 00:00", forecast_hours=[1, 6])
        mocker.patch.object(
            reaper,
            "_fetch_with_herbie",
            side_effect=APIError("herbie is not installed"),
        )
        with pytest.raises(APIError, match="herbie is not installed"):
            reaper.reap()

    def test_reap_mocked_success(self, mocker):
        """Test reap with mocked FastHerbie."""
        pytest.importorskip("herbie")

        # Create mock xarray Dataset that reflects genuine NWP structure with dims and coords
        mock_ds = _create_mock_hrrr_dataset(["tp", "t2m"])
        mocker.patch.object(mock_ds.herbie, "to_180")

        # Mock FastHerbie from herbie import
        mock_herbie = mocker.MagicMock()
        mock_herbie.return_value.xarray.return_value = mock_ds
        mocker.patch("herbie.fast.FastHerbie", mock_herbie)

        reaper = NWPReaper(init_time="2026-01-01 00:00", forecast_hours=[1, 6])

        harvested = reaper.reap()

        assert isinstance(harvested, xr.Dataset)
        assert len(harvested.data_vars) == 2
        assert "tp" in harvested.data_vars
        assert "t2m" in harvested.data_vars

    def test_reap_returns_harvested_data(self, mocker):
        """Test that reap returns properly formatted HarvestedData."""
        pytest.importorskip("herbie")

        mock_ds = _create_mock_hrrr_dataset(["tp"])
        mocker.patch.object(mock_ds.herbie, "to_180")

        mock_herbie = mocker.MagicMock()
        mock_herbie.return_value.xarray.return_value = mock_ds
        mocker.patch("herbie.fast.FastHerbie", mock_herbie)

        reaper = NWPReaper(init_time="2026-01-01 00:00", forecast_hours=[1, 6, 12])

        harvested = reaper.reap()

        # Check attrs
        assert harvested.attrs["model"] == "hrrr"
        assert harvested.attrs["product"] == "sfc"
        assert "tp" in harvested.data_vars

    def test_reap_api_error_handling(self, mocker):
        """Test that reap handles API errors gracefully."""
        pytest.importorskip("herbie")
        mock_herbie = mocker.MagicMock()
        mock_herbie.return_value.xarray.side_effect = Exception("Network error")
        mocker.patch("herbie.fast.FastHerbie", mock_herbie)

        reaper = NWPReaper(init_time="2026-01-01 00:00", forecast_hours=[1, 6])

        with pytest.raises(APIError, match="NWP fetch failed"):
            reaper.reap()


@pytest.mark.requires_herbie
class TestGetVariableNames:
    """Test variable extraction from HRRR data."""

    def test_variable_names_extracted(self, mocker):
        """Test that variable names are correctly extracted from Dataset."""
        pytest.importorskip("herbie")

        mock_ds = _create_mock_hrrr_dataset(["tp", "t2m", "u10"])
        mocker.patch.object(mock_ds.herbie, "to_180")

        mock_herbie = mocker.MagicMock()
        mock_herbie.return_value.xarray.return_value = mock_ds
        mocker.patch("herbie.fast.FastHerbie", mock_herbie)

        reaper = NWPReaper(init_time="2026-01-01 00:00", forecast_hours=[1])

        harvested = reaper.reap()

        assert len(harvested.data_vars) == 3
        assert set(harvested.data_vars.keys()) == {"tp", "t2m", "u10"}

    def test_empty_variables_handled(self, mocker):
        """Test handling of Dataset with no variables."""
        pytest.importorskip("herbie")

        mock_ds = _create_mock_hrrr_dataset([])  # Empty dataset
        mocker.patch.object(mock_ds.herbie, "to_180")

        mock_herbie = mocker.MagicMock()
        mock_herbie.return_value.xarray.return_value = mock_ds
        mocker.patch("herbie.fast.FastHerbie", mock_herbie)

        reaper = NWPReaper(init_time="2026-01-01 00:00", forecast_hours=[1])

        harvested = reaper.reap()
        assert len(harvested.data_vars) == 0
