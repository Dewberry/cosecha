"""Utility functions for data sowers, including transformations."""

from __future__ import annotations

import logging
from typing import Any, Optional, Union

import pandas as pd
import xarray as xr

from cosecha.sowing.exceptions import SowerError
from cosecha.data_models import HarvestedData

logger = logging.getLogger(__name__)


def apply_gridded_transformations(
    data: Union[HarvestedData, xr.Dataset], transformations: Optional[dict[str, Any]] = None
) -> Union[HarvestedData, xr.Dataset]:
    """Apply optional transformations to the xarray Dataset.

    Supported transformations:
    - 'spatial_subset': dict with 'lon_bounds' (min, max) and 'lat_bounds' (min, max)
    - 'unit_conversions': dict mapping variable names to conversion factors
    - 'variable_rename': dict mapping old variable names to new names
    - 'keep_variables': list of variables to keep

    Parameters
    ----------
    data : Union[HarvestedData, xr.Dataset]
        The harvested data or Dataset to transform.
    transformations : dict[str, Any], optional
        Dictionary of transformations to apply. If None, no transformations.

    Returns
    -------
    Union[HarvestedData, xr.Dataset]
        Harvested data or Dataset with transformations applied.

    Raises
    ------
    SowerError
        If transformations reference non-existent variables or coordinates.
    """
    if not transformations:
        return data

    is_harvested = isinstance(data, HarvestedData)
    dataset = data.data if is_harvested else data
    result = dataset.copy()

    # Spatial subset
    if "spatial_subset" in transformations:
        subset = transformations["spatial_subset"]
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
            raise SowerError(
                f"Spatial subset failed: {e}. Available dimensions: {list(ds.dims)}"
            )

    # Unit conversions
    if "unit_conversions" in transformations:
        conversions = transformations["unit_conversions"]
        for var, factor in conversions.items():
            if var not in result.data_vars:
                raise SowerError(
                    f"Variable '{var}' not found for unit conversion. "
                    f"Available: {list(result.data_vars)}"
                )
            result[var] = result[var] * factor
            logger.debug(f"Applied unit conversion to '{var}': factor={factor}")

    # Variable rename
    if "variable_rename" in transformations:
        renames = transformations["variable_rename"]
        result = result.rename(renames)
        if is_harvested:
            data.variable_names = [renames.get(var, var) for var in data.variable_names]
        logger.debug(f"Renamed variables: {renames}")

    # Keep specific variables
    if "keep_variables" in transformations:
        vars_to_keep = transformations["keep_variables"]
        missing = set(vars_to_keep) - set(result.data_vars)
        if missing:
            raise SowerError(
                f"Variables to keep not found: {missing}. "
                f"Available: {list(result.data_vars)}"
            )
        result = result[list(vars_to_keep)]
        if is_harvested:
            data.variable_names = [var for var in data.variable_names if var in vars_to_keep]
        logger.debug(f"Kept variables: {vars_to_keep}")

    if is_harvested:
        data.data = result
        return data

    return result

def apply_ts_transformations(
    data: Union[HarvestedData, pd.DataFrame], transformations: Optional[dict[str, Any]] = None
) -> Union[HarvestedData, pd.DataFrame]:
    """Apply optional transformations to the DataFrame.

    Supported transformations:
    - 'unit_conversions': dict mapping column names to conversion factors
    - 'rename_columns': dict mapping old column names to new names
    - 'filter_columns': list of columns to keep

    Parameters
    ----------
    data : Union[HarvestedData, pd.DataFrame]
        The harvested data or DataFrame to transform.
    transformations : dict[str, Any], optional
        Dictionary of transformations to apply. If None, no transformations.

    Returns
    -------
    Union[HarvestedData, pd.DataFrame]
        Transformed harvested data or DataFrame.

    Raises
    ------
    SowerError
        If transformations reference non-existent columns.
    """
    if not transformations:
        return data

    is_harvested = isinstance(data, HarvestedData)
    df = data.data if is_harvested else data
    result = df.copy()

    # Apply unit conversions
    if "unit_conversions" in transformations:
        conversions = transformations["unit_conversions"]
        for col, factor in conversions.items():
            if col not in result.columns:
                raise SowerError(
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
            raise SowerError(
                f"Columns {missing} not found for filtering. "
                f"Available: {list(result.columns)}"
            )
        result = result[cols]
        logger.debug(f"Filtered to columns: {cols}")

    if is_harvested:
        data.data = result
        return data
    return result
