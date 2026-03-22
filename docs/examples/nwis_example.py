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
# # NWIS Data Retrieval with Cosecha
#
# This notebook demonstrates how to fetch water resources data from the USGS National Water Information System (NWIS) using the `dataretrieval` library and process it with `cosecha`'s data handling utilities.
#
# **What you'll learn:**
# - Fetch instantaneous streamflow data from NWIS using the USGS `dataretrieval` library
# - Understand the structure of NWIS data
# - Wrap data in cosecha's `HarvestedData` object for standardized handling
# - Export data to Parquet format using cosecha's `ParquetSower`
# - Visualize water quality data over time

# %% [markdown]
# ## 1. Import Required Libraries
#
# We'll use several key libraries in this workflow:
# - **dataretrieval**: USGS-maintained library for accessing water data APIs
# - **cosecha**: Our data harvesting and storage library
# - **pandas**: Data manipulation and analysis
# - **matplotlib**: Time series visualization

# %%
from datetime import datetime
from pathlib import Path

# USGS water data retrieval
from dataretrieval import nwis

# Cosecha data handling
from cosecha import HarvestedData, ParquetSower

# Data analysis and visualization
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

print("✓ All libraries imported successfully")

# %% [markdown]
# ## 2. Initialize NWIS Data Source
#
# The **National Water Information System (NWIS)** is maintained by the USGS and provides real-time and historical water data from thousands of monitoring stations across the United States.
#
# ### About Instantaneous Data (`get_iv`)
#
# The `nwis.get_iv()` function retrieves *instantaneous* (real-time) measurements at their recorded frequency:
# - **5-minute intervals**: High-frequency stations (typical for flood monitoring)
# - **15-minute intervals**: Common for streamflow monitoring
# - **Hourly intervals**: Standard for many water quality parameters
# - **Other frequencies**: Varies by station and measurement type
#
# Parameter codes:
# - `00060`: Discharge (cubic feet per second)
# - `00065`: Gage height/stage (feet)
# - `00045`: Precipitation (inches)
# - Many other parameters available
#
# ### Advantages of Using `dataretrieval`
#
# ✓ **USGS-maintained**: Official library, handles API updates  
# ✓ **Clean DataFrames**: Returns ready-to-use pandas objects  
# ✓ **Metadata included**: Automatic parameter descriptions  
# ✓ **Error handling**: Robust retry logic and validation

# %% [markdown]
# ## 3. Query NWIS Data
#
# Let's fetch instantaneous streamflow data from a USGS monitoring station.
#
# **Station Details:**
# - **Site ID**: 08066500 - Trinity River at Dallas, TX
# - **Parameter**: 00060 - Discharge (cubic feet per second)
# - **Date Range**: January 2022 (example can be adjusted)

# %%
# Fetch instantaneous streamflow data from NWIS
print("Fetching instantaneous streamflow data from NWIS...")
print("Site: Trinity River at Dallas, TX (08066500)")
print("Parameter: Discharge (00060 - cubic feet per second)")
print("Date Range: Jan 1 - Jan 31, 2022\n")

df, metadata = nwis.get_iv(
    sites="08066500",           # USGS site ID
    start="2022-01-01",         # Start date (YYYY-MM-DD)
    end="2022-01-31",           # End date (YYYY-MM-DD)
    parameterCd="00060"         # Parameter code: Discharge
)

print(f"✓ Retrieved {len(df)} records")
print(f"✓ Date range: {df.index.min()} to {df.index.max()}")

# %% [markdown]
# ## 4. Process and Explore Results
#
# Let's examine the structure of the retrieved data and check for any data quality issues.

# %%
print("Data Structure:")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nIndex Type: {type(df.index).__name__} (timezone: {df.index.tz})")

# Display first few records
print("\n" + "="*70)
print("First 5 records:")
print("="*70)
df.head()

# %%
print("\n" + "="*70)
print("Data Quality Summary:")
print("="*70)
print(f"\nMissing values:\n{df.isnull().sum()}")

print("\n" + "="*70)
print("Streamflow Statistics (cubic feet per second):")
print("="*70)
# Get discharge column (parameter code 00060)
discharge_col = '00060'
if discharge_col in df.columns:
    print(df[discharge_col].describe())
    print(f"\nCharacteristics:")
    print(f"  Median flow: {df[discharge_col].median():.1f} cfs")
    print(f"  Peak flow: {df[discharge_col].max():.1f} cfs")
    print(f"  Min flow: {df[discharge_col].min():.1f} cfs")

# %% [markdown]
# ## 5. Wrap Data with Cosecha's HarvestedData
#
# The `HarvestedData` class standardizes how data is passed through the cosecha pipeline. It wraps the raw data with important metadata:
# - **Data**: The pandas DataFrame
# - **Source name**: Identifier for this data source
# - **Timestamp**: When the data was harvested
# - **Variable names**: Column names being harvested
# - **Metadata**: Additional context (source system, parameters, etc.)
#
# This standardization allows different "sowers" (output formats) to handle the data consistently.

# %%
# Create output directory if it doesn't exist
output_dir = Path("./data")
output_dir.mkdir(exist_ok=True)

# Wrap the data in HarvestedData
harvested = HarvestedData(
    data=df,
    source_name="USGS_Streamflow",
    timestamp=datetime.now(),
    variable_names=list(df.columns),
    metadata={
        "source": "NWIS",
        "sites": "08066500",
        "site_name": "Trinity River at Dallas, TX",
        "parameter": "00060",
        "parameter_name": "Discharge (cubic feet per second)",
        "data_type": "instantaneous",
    },
)

print("✓ Data wrapped in HarvestedData")
print(f"  Source: {harvested.source_name}")
print(f"  Records: {len(harvested.data)}")
print(f"  Variables: {harvested.variable_names}")

# %% [markdown]
# ## 6. Save to Parquet Format
#
# The `ParquetSower` writes the data to Apache Parquet format, which provides:
# - **Efficient compression**: Smaller file sizes than CSV
# - **Column-oriented storage**: Fast queries on specific fields
# - **Binary format**: Preserves data types and datetime precision
# - **Wide compatibility**: Supported by pandas, Apache Spark, and other tools
#
# The sower automatically generates timestamped filenames to avoid conflicts.

# %%
# Initialize the ParquetSower with output directory
sower = ParquetSower(output_dir=str(output_dir))

# Save the harvested data to Parquet
output_path = sower.sow(harvested)

print(f"✓ Data saved successfully!")
print(f"  Path: {output_path}")
print(f"  File size: {Path(output_path).stat().st_size:,} bytes")

# %% [markdown]
# ## 7. Read and Verify Saved Data
#
# Let's read the Parquet file back to verify everything was saved correctly.

# %%
# Read the Parquet file back
df_loaded = pd.read_parquet(output_path)

print("Verification:")
print(f"✓ Records loaded: {len(df_loaded)}")
print(f"✓ Columns: {list(df_loaded.columns)}")
print(f"✓ Date range preserved: {df_loaded.index.min()} to {df_loaded.index.max()}")
print(f"\nData types after loading:")
print(df_loaded.dtypes)
print(f"\nFirst 3 records from saved file:")
df_loaded.head(3)

# %% [markdown]
# ## 8. Visualize NWIS Data
#
# Let's create a time series plot of the streamflow data to visualize discharge patterns over the month.

# %%
fig, ax = plt.subplots(figsize=(14, 6))

discharge_col = '00060'
if discharge_col in df.columns:
    ax.plot(df.index, df[discharge_col], linewidth=1.5, color='steelblue', label='Discharge')
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Discharge (cubic feet per second)', fontsize=12)
    ax.set_title('Trinity River Streamflow - January 2022\n(Site 08066500: Dallas, TX)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    # Format x-axis dates
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.xticks(rotation=45)
    
    # Add statistics to plot
    mean_flow = df[discharge_col].mean()
    max_flow = df[discharge_col].max()
    ax.axhline(y=mean_flow, color='orange', linestyle='--', alpha=0.7, label=f'Mean: {mean_flow:.0f} cfs')
    
    # Format y-axis with thousands separator
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nFlow characteristics:")
    print(f"  Mean discharge: {mean_flow:.1f} cfs")
    print(f"  Peak discharge: {max_flow:.1f} cfs")
    print(f"  Min discharge: {df[discharge_col].min():.1f} cfs")
    print(f"  Std Dev: {df[discharge_col].std():.1f} cfs")

# %% [markdown]
# ## Summary
#
# You've successfully completed the NWIS data workflow! Here's what you learned:
#
# ### Key Takeaways
#
# 1. **Data Retrieval**: The `dataretrieval` library provides clean access to USGS NWIS data with automatic API handling
# 2. **Instantaneous Data**: `get_iv()` retrieves high-frequency measurements (5-min to hourly depending on station)
# 3. **Data Standardization**: `HarvestedData` wraps raw data with metadata for consistent handling
# 4. **Efficient Storage**: `ParquetSower` saves data in a compressed, queryable format
# 5. **Data Pipeline**: Fetch → Wrap → Save → Analyze (as demonstrated here)
#
# ### Next Steps
#
# - **Multiple parameters**: Query discharge (00060), stage (00065), or precipitation (00045) together
# - **Multiple sites**: Fetch data from multiple USGS stations in one call by using comma-separated site IDs
# - **Different writers**: Replace `ParquetSower` with `ZarrSower`, `NetCDFSower`, or other output formats
# - **Automation**: Schedule this notebook to run daily for continuous data collection
# - **Advanced analysis**: Combine with xarray for multi-dimensional analysis of gridded data
#
# ### Resources
#
# - [USGS NWIS Web Services](https://waterservices.usgs.gov/)
# - [dataretrieval Documentation](https://github.com/DOI-USGS/dataretrieval-python)
# - [Cosecha Documentation](https://dewberry.github.io/cosecha/)
# - [Apache Parquet Format](https://parquet.apache.org/)
