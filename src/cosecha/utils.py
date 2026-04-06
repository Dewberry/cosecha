"""Utility functions for data sowers, including transformations."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

from cosecha.exceptions import TransformationError
from cosecha.logging import logger

if TYPE_CHECKING:
    from collections.abc import Generator

    import pandas as pd
    import xarray as xr


@contextlib.contextmanager
def wrap_errors(
    exc_type: type[Exception],
    message: str,
    *passthrough: type[Exception],
) -> Generator[None, None, None]:
    """Context manager that logs and re-wraps unexpected exceptions.

    Exceptions listed in *passthrough are re-raised as-is; everything else
    is logged and re-raised as exc_type.
    """
    try:
        yield
    except BaseException as e:
        if passthrough and isinstance(e, passthrough):
            raise
        logger.exception(message)
        raise exc_type(f"{message}: {e}") from e


def _apply_spatial_subset(result: xr.Dataset, subset: dict[str, Any]) -> xr.Dataset:
    try:
        if "lon_bounds" in subset or "lat_bounds" in subset:
            # Handle naming variances (usually latitude/longitude for Herbie, lat/lon for generic)
            lat_coord = result.latitude if hasattr(result, "latitude") else result.lat
            lon_coord = result.longitude if hasattr(result, "longitude") else result.lon

            if result.coords[lat_coord.name].ndim > 1 or result.coords[lon_coord.name].ndim > 1:
                mask = None

                if "lat_bounds" in subset:
                    lat_min, lat_max = subset["lat_bounds"]
                    lat_mask = (lat_coord >= lat_min) & (lat_coord <= lat_max)
                    mask = lat_mask

                if "lon_bounds" in subset:
                    lon_min, lon_max = subset["lon_bounds"]
                    lon_mask = (lon_coord >= lon_min) & (lon_coord <= lon_max)
                    mask = lon_mask if mask is None else (mask & lon_mask)

                if mask is not None:
                    # Find the bounding box in y/x index space
                    y_idx, x_idx = mask.values.nonzero()

                    if len(y_idx) > 0 and len(x_idx) > 0:
                        y_slice = slice(y_idx.min(), y_idx.max() + 1)
                        x_slice = slice(x_idx.min(), x_idx.max() + 1)
                        result = result.isel(y=y_slice, x=x_slice)
                        logger.debug(
                            f"Applied bounding box subset: lat={subset.get('lat_bounds')}, "
                            f"lon={subset.get('lon_bounds')}"
                        )
                    else:
                        logger.warning("Spatial subset resulted in an empty mask; skipping subset")
            else:
                result = result.sel(
                    {
                        lat_coord.name: slice(*subset.get("lat_bounds", (None, None))),
                        lon_coord.name: slice(*subset.get("lon_bounds", (None, None))),
                    }
                )
        else:
            logger.warning("spatial_subset requires at least one of 'lon_bounds' or 'lat_bounds'")

    except Exception as e:
        raise TransformationError(
            f"Spatial subset failed: {e}. Available dimensions: {list(result.dims)}"
        ) from e
    return result


def apply_gridded_transformations(
    data: xr.Dataset, transformations: dict[str, Any] | None = None
) -> xr.Dataset:
    """Apply optional transformations to the xarray Dataset.

    Supported transformations:
    - 'spatial_subset': dict with 'lon_bounds' (min, max) and 'lat_bounds' (min, max)
    - 'unit_conversions': dict mapping variable names to conversion factors
    - 'variable_rename': dict mapping old variable names to new names
    - 'keep_variables': list of variables to keep

    Parameters
    ----------
    data : xr.Dataset
        The Dataset to transform.
    transformations : dict[str, Any], optional
        Dictionary of transformations to apply. If None, no transformations.

    Returns
    -------
    xr.Dataset
        Dataset with transformations applied.

    Raises
    ------
    TransformationError
        If transformations reference non-existent variables or coordinates.
    """
    if not transformations:
        return data

    result = data.copy()

    # Spatial subset
    if "spatial_subset" in transformations:
        result = _apply_spatial_subset(result, transformations["spatial_subset"])

    # Unit conversions
    if "unit_conversions" in transformations:
        conversions = transformations["unit_conversions"]
        for var, factor in conversions.items():
            if var not in result.data_vars:
                raise TransformationError(
                    f"Variable '{var}' not found for unit conversion. "
                    f"Available: {list(result.data_vars)}"
                )
            result[var] = result[var] * factor
            logger.debug(f"Applied unit conversion to '{var}': factor={factor}")

    # Variable rename
    if "variable_rename" in transformations:
        renames = transformations["variable_rename"]
        result = result.rename(renames)
        logger.debug(f"Renamed variables: {renames}")

    # Keep specific variables
    if "keep_variables" in transformations:
        vars_to_keep = transformations["keep_variables"]
        missing = set(vars_to_keep) - set(result.data_vars)
        if missing:
            raise TransformationError(
                f"Variables to keep not found: {missing}. Available: {list(result.data_vars)}"
            )
        result = result[list(vars_to_keep)]
        logger.debug(f"Kept variables: {vars_to_keep}")

    return result


def apply_ts_transformations(
    data: pd.DataFrame, transformations: dict[str, Any] | None = None
) -> pd.DataFrame:
    """Apply optional transformations to the DataFrame.

    Supported transformations:
    - 'unit_conversions': dict mapping column names to conversion factors
    - 'rename_columns': dict mapping old column names to new names
    - 'filter_columns': list of columns to keep

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to transform.
    transformations : dict[str, Any], optional
        Dictionary of transformations to apply. If None, no transformations.

    Returns
    -------
    pd.DataFrame
        Transformed DataFrame.

    Raises
    ------
    TransformationError
        If transformations reference non-existent columns.
    """
    if not transformations:
        return data

    result = data.copy()

    # Apply unit conversions
    if "unit_conversions" in transformations:
        conversions = transformations["unit_conversions"]
        for col, factor in conversions.items():
            if col not in result.columns:
                raise TransformationError(
                    f"Column '{col}' not found for unit conversion. "
                    f"Available: {list(result.columns)}"
                )
            result[col] = result[col] * factor
            logger.debug(f"Applied unit conversion to '{col}': factor={factor}")

    # Rename columns
    if "rename_columns" in transformations:
        renames = transformations["rename_columns"]
        result = result.rename(columns=renames)
        logger.debug(f"Renamed columns: {renames}")

    # Filter to specific columns
    if "filter_columns" in transformations:
        cols = transformations["filter_columns"]
        missing = set(cols) - set(result.columns)
        if missing:
            raise TransformationError(
                f"Columns {missing} not found for filtering. Available: {list(result.columns)}"
            )
        result = result[cols]
        logger.debug(f"Filtered to columns: {cols}")

    return result
