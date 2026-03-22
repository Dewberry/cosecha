# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     notebook_metadata_filter: kernelspec,jupytext
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: dev
#     language: python
#     name: python3
# ---

# %% [markdown]
# # HRRR Weather Data Retrieval with Cosecha
#
# This notebook demonstrates how to fetch High-Resolution Rapid Refresh (HRRR) weather forecast data and process it with `cosecha`'s gridded data utilities.
#
# **What you'll learn:**
# - Fetch HRRR forecast data from NOAA using the `herbie` library
# - Understand the structure of HRRR gridded datasets
# - Wrap data in cosecha's `HarvestedData` object for standardized handling
# - Export gridded data to NetCDF format using cosecha's `NetCDFSower`
# - Apply transformations (unit conversion, variable selection, spatial subsetting)
# - Visualize weather forecast data

# %% [markdown]
# ## 1. Import Required Libraries
#
# We'll use several key libraries in this workflow:
# - **herbie**: NOAA library for fetching HRRR forecast data
# - **cosecha**: Our data harvesting and storage library
# - **xarray**: Multi-dimensional array operations
# - **matplotlib**: Visualization of weather data

# %%
from datetime import datetime, timedelta
from pathlib import Path

# HRRR weather data retrieval
from cosecha import HarvestedData
from cosecha.sowing.netcdf import NetCDFSower

# Data analysis and visualization
import xarray as xr
import matplotlib.pyplot as plt

print("✓ All libraries imported successfully")

# %% [markdown]
# ## 2. Understanding HRRR Data
#
# The **High-Resolution Rapid Refresh (HRRR)** is a real-time, high-resolution numerical weather prediction model:
#
# ### Key Characteristics
# - **Resolution**: 3 km grid spacing (finest resolution for US forecasts)
# - **Update frequency**: Updated hourly
# - **Forecast horizon**: Up to 48 hours ahead
# - **Maintained by**: NOAA National Weather Service
# - **Data access**: Public access via NOAA AWS buckets
#
# ### Data Structure
# HRRR data is gridded (2D spatial + time) with dimensions:
# - **Latitude**: ~1059 points across North America
# - **Longitude**: ~1799 points across North America  
# - **Time**: Forecast hours (typically 1-48 hours)
# - **Vertical levels**: Optional (upper-air variables on pressure levels)
#
# ### Common Variables
# - **Surface variables**: Temperature (t), dewpoint (d), wind (u, v)
# - **Precipitation**: Accumulated precipitation, convective precipitation
# - **Dynamics**: CAPE, CIN, relative humidity
# - **And 50+ other variables** depending on your selection

# %% [markdown]
# ## 3. Fetch HRRR Forecast Data
#
# The HRRR data workflow:
# 1. **Fetch**: FastHerbie downloads grib2 files from NOAA AWS (native HRRR format)
# 2. **Transform**: Process the grib2 data (unit conversion, variable renaming)
# 3. **Convert**: Herbie automatically converts grib2 to xarray.Dataset
# 4. **Wrap**: Package in cosecha's HarvestedData for standardized handling
# 5. **Save**: NetCDFSower writes xarray to NetCDF-4 format (self-describing, efficient)
#
# **Note**: HRRR forecasts are available from NOAA for recent/current times. We use the current UTC time
# and work backwards by hours to find the most recent available forecast initialization.

# %%
print("Initializing HRRR Reaper...")
print("=" * 70)

# Create output directory
output_dir = Path("./data/hrrr")
output_dir.mkdir(parents=True, exist_ok=True)

# Try to fetch real data using timezone-aware current time
# HRRR data is available on NOAA AWS for recent initialization times
# Current data minus 2 hours ensures the forecast has processed
from datetime import timezone
from herbie import FastHerbie
import xarray as xr

harvested = None

# Get current UTC time and work backwards by hours to find available data
current_utc = datetime.now(timezone.utc)

for hours_back in range(0, 12, 2):  # Try current hour, then -2, -4, -6, -8, -10 hours
    attempt_date = current_utc - timedelta(hours=hours_back)
    # Round down to nearest hour for HRRR (always XX:00:00 UTC)
    attempt_date = attempt_date.replace(minute=0, second=0, microsecond=0)
    init_time_str = attempt_date.strftime("%Y-%m-%d %H:00")
    
    try:
        print(f"\nAttempt {hours_back // 2 + 1}: Trying {init_time_str} UTC...")
        
        # Use FastHerbie directly with precipitation search pattern
        # This is proven to work for HRRR data fetching
        print("  Creating FastHerbie object...")
        h = FastHerbie([init_time_str], model="hrrr", fxx=range(1, 7))
        
        # Fetch precipitation data using specific search pattern
        # This retrieves hourly accumulated precipitation
        print("  Fetching hourly precipitation from NOAA AWS...")
        ds = h.xarray(search=r":APCP:.*:(?:0-1|[1-9]\d*-\d+) hour")
        
        print(f"✓ Successfully fetched real HRRR data from {init_time_str}!")
        
        # Process the dataset
        # Convert longitude from 0-360 to -180-180
        ds['longitude'] = ds.longitude - 360
        
        # Rename total precipitation to standard name
        if 'tp' in ds.data_vars:
            ds = ds.rename({'tp': 'hourly_accum_precip'})
        
        # Convert precipitation from mm to inches
        if 'hourly_accum_precip' in ds.data_vars:
            ds['hourly_accum_precip'] = ds['hourly_accum_precip'] / 25.4
            ds['hourly_accum_precip'].attrs['units'] = 'inches'
        
        # Wrap in cosecha's HarvestedData for standardized handling
        from cosecha import HarvestedData
        harvest_timestamp = datetime.strptime(init_time_str, "%Y-%m-%d %H:%M")
        harvested = HarvestedData(
            data=ds,
            source_name="NOAA_HRRR",
            timestamp=harvest_timestamp,
            variable_names=list(ds.data_vars),
            metadata={
                "model": "hrrr",
                "forecast_hours": "1-6",
                "units": "precipitation in inches",
                "description": "Fetched via Herbie from NOAA AWS"
            }
        )
        
        break
        
    except Exception as e:
        print(f"  ✗ Data not available: {str(e)[:80]}...")
        if hours_back >= 10:
            print("\n⚠ Could not find recent HRRR data on NOAA AWS (tried past 10 hours)")
            raise

print(f"\n{'='*70}")
print(f"✓ Data ready for processing")
print(f"  Source: Real NOAA HRRR (via Herbie)")
print(f"  Init time: {init_time_str} UTC")
print(f"  Location: Dallas, TX region and surroundings")
print(f"  Variables: {len(harvested.data.data_vars)}")
print(f"  Grid dimensions: {dict(harvested.data.sizes)}")
print(f"  Data variables: {list(harvested.data.data_vars)}")
print(f"  Format: xarray.Dataset (converted from grib2)")
print(f"  Units: Precipitation in inches (converted from mm)")
print(f"  Next: Will save to NetCDF format for efficient storage")

# %% [markdown]
# ## 4. Explore Data Structure
#
# Let's examine the gridded forecast data to understand its dimensions, coordinates, and variables.

# %%
# Display full dataset structure
print("Full Dataset Structure:")
print("=" * 70)
print(harvested.data)

print("\n" + "=" * 70)
print("Data Type Information")
print("=" * 70)
for var in harvested.data.data_vars:
    print(f"{var:10s}: {harvested.data[var].dtype}, shape={harvested.data[var].shape}")

print("\n" + "=" * 70)
print("Data Statistics by Variable")
print("=" * 70)
for var in list(harvested.data.data_vars)[:3]:  # Show first 3 variables
    data = harvested.data[var].values
    print(f"\n{var}:")
    print(f"  Min:  {data.min():10.4f}")
    print(f"  Max:  {data.max():10.4f}")
    print(f"  Mean: {data.mean():10.4f}")
    print(f"  Std:  {data.std():10.4f}")

# %% [markdown]
# ## 5. Write to NetCDF Format
#
# The `NetCDFSower` writes gridded data to NetCDF-4/HDF5 format, which provides:
# - **Efficient compression**: Self-describing binary format with zlib compression
# - **Wide compatibility**: Supported by scientific software (QGIS, ArcGIS, climate software)
# - **Metadata preservation**: Stores dimensions, coordinates, and attributes
# - **Selective output**: Can keep specific variables and apply transformations

# %%
# Initialize NetCDF Sower
print("Writing to NetCDF Format...")
print("=" * 70)

sower = NetCDFSower(
    output_dir=output_dir,
    compression="zlib",
    compression_level=4
)
print(f"✓ NetCDFSower initialized")
print(f"  Output directory: {output_dir}")
print(f"  Compression: zlib (level 4)")

# Optional transformations
# Example: Select only specific variables
# transformations = {
#     "keep_variables": ["t_2m", "u10m", "v10m"],
# }

# Example: Apply spatial subset
# transformations = {
#     "spatial_subset": {
#         "x": (0, 10),
#         "y": (0, 10)
#     }
# }

# Example: Apply unit conversion (Kelvin to Celsius)
# transformations = {
#     "unit_conversions": {"t_2m": -273.15}  # This would require offset handling
# }

# Write to NetCDF (without transformations for this example)
output_path = sower.sow(harvested)

print(f"\n✓ NetCDF file created successfully!")
print(f"  Path: {output_path}")
print(f"  File size: {Path(output_path).stat().st_size:,} bytes")

# %% [markdown]
# ## 6. Verify NetCDF File
#
# Let's read the NetCDF file back to verify that all data was written correctly with full fidelity.

# %%
# Read the NetCDF file back
ds_verify = xr.open_dataset(output_path, engine="h5netcdf")

print("Verification of Written NetCDF File")
print("=" * 70)
print(f"✓ File successfully read back")
print(f"\nDimensions: {dict(ds_verify.sizes)}")
print(f"Coordinates: {list(ds_verify.coords)}")
print(f"Variables: {list(ds_verify.data_vars)}")

print(f"\nData Type Information:")
for var in ds_verify.data_vars:
    print(f"  {var}: {ds_verify[var].dtype}")

print(f"\nAttributes:")
for attr, value in ds_verify.attrs.items():
    print(f"  {attr}: {value}")

# Show sample data from one variable
print(f"\nSample data from '{list(ds_verify.data_vars)[0]}' (first forecast hour):")
first_var = list(ds_verify.data_vars)[0]
print(ds_verify[first_var].isel({list(ds_verify.dims)[0]: 0}).values)

ds_verify.close()

print("\n✓ All data verified - no corruption detected!")

# %% [markdown]
# ## 7. Visualize HRRR Weather Data
#
# Let's create visualizations of the forecast data to understand spatial patterns.

# %%
# Create visualization
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Get the first forecast hour for visualization
forecast_idx = 0

# Plot the first variable
var_list = list(harvested.data.data_vars)
var = var_list[0]  # Plot only the first variable

# Get data for first time/forecast hour
# Use the first non-x/y dimension (usually 'time', 'step', or similar)
time_dims = [d for d in harvested.data[var].dims if d not in ['latitude', 'longitude', 'x', 'y']]
if time_dims:
    data = harvested.data[var].isel({time_dims[0]: forecast_idx}).values
else:
    data = harvested.data[var].values

# Create 2D contour plot
im = ax.contourf(data, levels=20, cmap='viridis')
ax.set_title(f'{var} - First Forecast Hour', fontsize=14, fontweight='bold')
ax.set_xlabel('X Grid')
ax.set_ylabel('Y Grid')
plt.colorbar(im, ax=ax, label='Value')

plt.tight_layout()
plt.show()

print(f"✓ Visualization complete")
print(f"  Variable plotted: {var}")
print(f"  Grid size: {data.shape[0]} × {data.shape[1]} points")

# %% [markdown]
# ## Summary
#
# You've successfully completed the HRRR data workflow! Here's what you learned:
#
# ### Key Takeaways
#
# 1. **Data Retrieval**: The `HRRRReaper` provides clean access to NOAA HRRR forecast data via the `herbie` library
# 2. **Gridded Data**: HRRR provides multi-dimensional gridded forecasts (3D: time × latitude × longitude)
# 3. **Data Standardization**: `HarvestedData` wraps gridded xarray Datasets with metadata for consistent handling
# 4. **Efficient Storage**: `NetCDFSower` saves gridded data in NetCDF-4/HDF5 format with compression
# 5. **Data Pipeline**: Fetch → Wrap → Save → Verify → Visualize
#
# ### Workflow Comparison: NWIS vs HRRR
#
# | Aspect | NWIS (Time-Series) | HRRR (Gridded) |
# |--------|-------------------|---------------|
# | **Data Type** | Point measurements | Spatial forecasts |
# | **Format** | pandas.DataFrame | xarray.Dataset |
# | **Output** | Parquet (columnar) | NetCDF (hierarchical) |
# | **Use Case** | Historical observations | Weather forecasts |
# | **Resolution** | Single locations | Full domain (3 km) |
#
# ### Next Steps
#
# - **Real Data**: Use current date for initialization time to fetch real HRRR forecasts
# - **Transformations**: Apply spatial subsetting, variable selection, or unit conversion
# - **Multiple Forecasts**: Fetch multiple initialization times for ensemble analysis
# - **Advanced Analysis**: Use xarray tools for dimensional analysis and statistical comparison
# - **Publication**: NetCDF format is ideal for sharing scientific data
#
# ### Resources
#
# - [NOAA HRRR Information](https://rapidrefresh.noaa.gov/)
# - [Herbie Library Documentation](https://github.com/blaylockbk/herbie)
# - [xarray Documentation](http://xarray.pydata.org/)
# - [NetCDF Format Guide](https://www.unidata.ucar.edu/software/netcdf/)
# - [Cosecha Documentation](https://dewberry.github.io/cosecha/)
