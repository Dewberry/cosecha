"""Tests for utility functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from cosecha.exceptions import TransformationError
from cosecha.utils import (
    apply_gridded_transformations,
    apply_ts_transformations,
    to_180,
    wrap_errors,
)


class TestWrapErrors:
    """Tests for wrap_errors context manager."""

    def test_no_exception(self):
        """Test context manager completes normally when no exception is raised."""
        with wrap_errors(RuntimeError, "should not fire"):
            result = 1 + 1
        assert result == 2

    def test_exception_wrapped(self):
        """Test that a raised exception is wrapped in the specified type."""
        with pytest.raises(ValueError, match="operation failed: something broke"):
            with wrap_errors(ValueError, "operation failed"):
                raise RuntimeError("something broke")

    def test_wrapped_exception_chains_original(self):
        """Test that the original exception is chained via __cause__."""
        with pytest.raises(ValueError) as exc_info:
            with wrap_errors(ValueError, "wrapper"):
                raise KeyError("original")
        assert isinstance(exc_info.value.__cause__, KeyError)

    def test_passthrough_exception_not_wrapped(self):
        """Test that passthrough exceptions are re-raised as-is."""
        with pytest.raises(KeyboardInterrupt):
            with wrap_errors(RuntimeError, "should not wrap", KeyboardInterrupt):
                raise KeyboardInterrupt

    def test_passthrough_list_non_matching_still_wrapped(self):
        """Test that non-matching exceptions are still wrapped even with passthrough list."""
        with pytest.raises(ValueError, match="wrapped: oops"):
            with wrap_errors(ValueError, "wrapped", KeyboardInterrupt, SystemExit):
                raise RuntimeError("oops")


class TestTo180:
    """Tests for to_180 longitude conversion."""

    def test_1d_longitude_converted_and_sorted(self):
        """Test 1D longitude [0,360] is converted to [-180,180] and sorted."""
        lons = np.array([0, 90, 180, 270, 350])
        da = xr.DataArray(
            np.arange(5),
            dims=["longitude"],
            coords={"longitude": lons},
        )
        result = to_180(da)
        expected_lons = np.array([-180, -90, -10, 0, 90])
        np.testing.assert_array_equal(result.longitude.values, expected_lons)

    def test_1d_longitude_already_180_no_change(self):
        """Test 1D longitude already in [-180,180] remains unchanged and sorted."""
        lons = np.array([-180, -90, 0, 90, 179])
        ds = xr.Dataset(
            {"temp": (["longitude"], np.arange(5))},
            coords={"longitude": lons},
        )
        result = to_180(ds)
        expected_lons = np.array([-180, -90, 0, 90, 179])
        np.testing.assert_array_equal(result.longitude.values, expected_lons)

    def test_2d_longitude_converted_not_sorted(self):
        """Test 2D (curvilinear) longitude is converted but not reordered."""
        lons_2d = np.array([[350, 10], [190, 200]])
        lats_2d = np.array([[40, 41], [42, 43]])
        ds = xr.Dataset(
            {"temp": (["y", "x"], np.ones((2, 2)))},
            coords={
                "longitude": (["y", "x"], lons_2d),
                "latitude": (["y", "x"], lats_2d),
            },
        )
        result = to_180(ds)
        expected_lons = np.array([[-10, 10], [-170, -160]])
        np.testing.assert_array_equal(result.longitude.values, expected_lons)

    def test_works_on_dataarray(self):
        """Test to_180 works on xarray DataArray."""
        lons = np.array([0, 180, 270])
        da = xr.DataArray(
            [1, 2, 3],
            dims=["longitude"],
            coords={"longitude": lons},
        )
        result = to_180(da)
        assert isinstance(result, xr.DataArray)
        np.testing.assert_array_equal(result.longitude.values, [-180, -90, 0])

    def test_works_on_dataset(self):
        """Test to_180 works on xarray Dataset."""
        lons = np.array([0, 180, 270])
        ds = xr.Dataset(
            {"var": (["longitude"], [1, 2, 3])},
            coords={"longitude": lons},
        )
        result = to_180(ds)
        assert isinstance(result, xr.Dataset)
        np.testing.assert_array_equal(result.longitude.values, [-180, -90, 0])

    def test_does_not_mutate_input(self):
        """Test that to_180 does not modify the original data."""
        lons = np.array([0, 90, 270])
        da = xr.DataArray([1, 2, 3], dims=["longitude"], coords={"longitude": lons})
        _ = to_180(da)
        np.testing.assert_array_equal(da.longitude.values, lons)


class TestApplyGriddedTransformations:
    """Tests for apply_gridded_transformations."""

    @pytest.fixture()
    def sample_ds(self):
        """Create a sample Dataset with 1D lat/lon coordinates."""
        return xr.Dataset(
            {
                "temperature": (["latitude", "longitude"], np.arange(12).reshape(3, 4).astype(float)),
                "pressure": (["latitude", "longitude"], np.ones((3, 4))),
            },
            coords={
                "latitude": [10.0, 20.0, 30.0],
                "longitude": [-100.0, -90.0, -80.0, -70.0],
            },
        )

    @pytest.fixture()
    def curvilinear_ds(self):
        """Create a sample Dataset with 2D lat/lon coordinates (curvilinear)."""
        lats = np.array([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]])
        lons = np.array([[-100.0, -90.0], [-100.0, -90.0], [-100.0, -90.0]])
        return xr.Dataset(
            {
                "temperature": (["y", "x"], np.arange(6).reshape(3, 2).astype(float)),
            },
            coords={
                "latitude": (["y", "x"], lats),
                "longitude": (["y", "x"], lons),
            },
        )

    def test_none_transformations_returns_unchanged(self, sample_ds):
        """Test that None transformations returns data as-is."""
        result = apply_gridded_transformations(sample_ds, None)
        xr.testing.assert_identical(result, sample_ds)

    def test_empty_transformations_returns_unchanged(self, sample_ds):
        """Test that empty dict transformations returns data as-is."""
        result = apply_gridded_transformations(sample_ds, {})
        xr.testing.assert_identical(result, sample_ds)

    def test_spatial_subset_1d(self, sample_ds):
        """Test spatial subset with 1D lat/lon coordinates."""
        transforms = {"spatial_subset": {"lat_bounds": (15.0, 25.0), "lon_bounds": (-95.0, -75.0)}}
        result = apply_gridded_transformations(sample_ds, transforms)
        assert result.latitude.values.tolist() == [20.0]
        assert result.longitude.values.tolist() == [-90.0, -80.0]

    def test_spatial_subset_2d_curvilinear(self, curvilinear_ds):
        """Test spatial subset with 2D lat/lon coordinates (curvilinear grid)."""
        transforms = {"spatial_subset": {"lat_bounds": (15.0, 35.0), "lon_bounds": (-105.0, -85.0)}}
        result = apply_gridded_transformations(curvilinear_ds, transforms)
        assert "temperature" in result.data_vars
        assert result.sizes["y"] == 2
        assert result.sizes["x"] == 2

    def test_spatial_subset_2d_empty_mask_logs_warning(self, curvilinear_ds, caplog):
        """Test spatial subset with 2D coords and no matching points logs warning."""
        transforms = {"spatial_subset": {"lat_bounds": (90.0, 91.0)}}
        result = apply_gridded_transformations(curvilinear_ds, transforms)
        assert result.sizes == curvilinear_ds.sizes

    def test_spatial_subset_no_bounds_returns_unchanged(self, sample_ds):
        """Test spatial subset with no bounds keys returns data unchanged."""
        transforms = {"spatial_subset": {"other_key": "value"}}
        result = apply_gridded_transformations(sample_ds, transforms)
        xr.testing.assert_identical(result, sample_ds)

    def test_unit_conversions(self, sample_ds):
        """Test unit conversions multiply variables by factor."""
        transforms = {"unit_conversions": {"temperature": 1.8, "pressure": 100.0}}
        result = apply_gridded_transformations(sample_ds, transforms)
        np.testing.assert_allclose(result["temperature"].values, sample_ds["temperature"].values * 1.8)
        np.testing.assert_allclose(result["pressure"].values, sample_ds["pressure"].values * 100.0)

    def test_unit_conversions_missing_variable(self, sample_ds):
        """Test unit conversions raises TransformationError for missing variable."""
        transforms = {"unit_conversions": {"nonexistent": 2.0}}
        with pytest.raises(TransformationError, match="nonexistent"):
            apply_gridded_transformations(sample_ds, transforms)

    def test_variable_rename(self, sample_ds):
        """Test variable rename renames data variables."""
        transforms = {"variable_rename": {"temperature": "temp_k"}}
        result = apply_gridded_transformations(sample_ds, transforms)
        assert "temp_k" in result.data_vars
        assert "temperature" not in result.data_vars

    def test_keep_variables(self, sample_ds):
        """Test keep_variables filters to only listed variables."""
        transforms = {"keep_variables": ["temperature"]}
        result = apply_gridded_transformations(sample_ds, transforms)
        assert list(result.data_vars) == ["temperature"]

    def test_keep_variables_missing_raises(self, sample_ds):
        """Test keep_variables raises TransformationError for missing variable."""
        transforms = {"keep_variables": ["nonexistent"]}
        with pytest.raises(TransformationError, match="nonexistent"):
            apply_gridded_transformations(sample_ds, transforms)

    def test_multiple_transformations(self, sample_ds):
        """Test multiple transformations applied together in order."""
        transforms = {
            "unit_conversions": {"temperature": 2.0},
            "variable_rename": {"temperature": "temp_scaled"},
            "keep_variables": ["temp_scaled"],
        }
        result = apply_gridded_transformations(sample_ds, transforms)
        assert list(result.data_vars) == ["temp_scaled"]
        np.testing.assert_allclose(result["temp_scaled"].values, sample_ds["temperature"].values * 2.0)


class TestApplyTsTransformations:
    """Tests for apply_ts_transformations."""

    @pytest.fixture()
    def sample_df(self):
        """Create a sample DataFrame."""
        return pd.DataFrame(
            {
                "flow": [1.0, 2.0, 3.0],
                "level": [10.0, 20.0, 30.0],
                "quality": ["A", "B", "C"],
            }
        )

    def test_none_transformations_returns_unchanged(self, sample_df):
        """Test that None transformations returns data as-is."""
        result = apply_ts_transformations(sample_df, None)
        pd.testing.assert_frame_equal(result, sample_df)

    def test_empty_transformations_returns_unchanged(self, sample_df):
        """Test that empty dict transformations returns data as-is."""
        result = apply_ts_transformations(sample_df, {})
        pd.testing.assert_frame_equal(result, sample_df)

    def test_unit_conversions(self, sample_df):
        """Test unit conversions multiply columns by factor."""
        transforms = {"unit_conversions": {"flow": 0.0283168}}
        result = apply_ts_transformations(sample_df, transforms)
        np.testing.assert_allclose(result["flow"].values, sample_df["flow"].values * 0.0283168)
        np.testing.assert_array_equal(result["level"].values, sample_df["level"].values)

    def test_unit_conversions_missing_column(self, sample_df):
        """Test unit conversions raises TransformationError for missing column."""
        transforms = {"unit_conversions": {"nonexistent": 2.0}}
        with pytest.raises(TransformationError, match="nonexistent"):
            apply_ts_transformations(sample_df, transforms)

    def test_rename_columns(self, sample_df):
        """Test rename_columns renames DataFrame columns."""
        transforms = {"rename_columns": {"flow": "discharge", "level": "stage"}}
        result = apply_ts_transformations(sample_df, transforms)
        assert "discharge" in result.columns
        assert "stage" in result.columns
        assert "flow" not in result.columns
        assert "level" not in result.columns

    def test_filter_columns(self, sample_df):
        """Test filter_columns keeps only listed columns."""
        transforms = {"filter_columns": ["flow", "quality"]}
        result = apply_ts_transformations(sample_df, transforms)
        assert list(result.columns) == ["flow", "quality"]

    def test_filter_columns_missing_raises(self, sample_df):
        """Test filter_columns raises TransformationError for missing column."""
        transforms = {"filter_columns": ["flow", "nonexistent"]}
        with pytest.raises(TransformationError, match="nonexistent"):
            apply_ts_transformations(sample_df, transforms)

    def test_multiple_transformations(self, sample_df):
        """Test multiple transformations applied together in order."""
        transforms = {
            "unit_conversions": {"flow": 2.0},
            "rename_columns": {"flow": "discharge_scaled"},
            "filter_columns": ["discharge_scaled"],
        }
        result = apply_ts_transformations(sample_df, transforms)
        assert list(result.columns) == ["discharge_scaled"]
        np.testing.assert_allclose(result["discharge_scaled"].values, sample_df["flow"].values * 2.0)

    def test_does_not_mutate_input(self, sample_df):
        """Test that transformations do not modify the original DataFrame."""
        original_flow = sample_df["flow"].values.copy()
        transforms = {"unit_conversions": {"flow": 100.0}}
        apply_ts_transformations(sample_df, transforms)
        np.testing.assert_array_equal(sample_df["flow"].values, original_flow)
