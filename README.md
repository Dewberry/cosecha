# Cosecha

Tools for harvesting earth observation data for use in flood forecasting.

Cosecha provides a flexible pipeline for collecting geospatial data from multiple
sources and writing to various formats with optional transformations.

## Features

- Time-series data collection (USGS NWIS, IEM ASOS, NWS Local Storm Reports, USACE
    reservoirs)
- Gridded data support (HRRR, RRFS, RTMA via herbie; MRMS via S3; WPC 6-hour QPF)
- Multiple output formats: Parquet, NetCDF, Zarr, Iceberg, IceChunk
- Data transformations: unit conversion, spatial subsetting, variable selection/rename
- Cross-platform support (ecCodes C library required for GRIB2/MRMS)

## Available Reapers

| Reaper            | Type        | Source                                                        |
| ----------------- | ----------- | ------------------------------------------------------------- |
| `USGSNWISReaper`  | Time series | USGS NWIS streamflow, stage and precipitation                 |
| `ASOSReaper`      | Time series | ASOS surface observations from the Iowa Environmental Mesonet |
| `LSRReaper`       | Time series | NWS Local Storm Reports from the Iowa Environmental Mesonet   |
| `ReservoirReaper` | Time series | USACE CDA reservoir storage, elevation and outflow            |
| `NWPReaper`       | Gridded     | HRRR, RRFS, RTMA and other NWP models via herbie (`[nwp]`)    |
| `MRMSReaper`      | Gridded     | NOAA MRMS accumulated precipitation from S3                   |
| `WPCQPFReaper`    | Gridded     | NOAA WPC 2.5 km CONUS 6-hour QPF forecasts                    |

## Installation

```console
pip install cosecha
```

With optional dependencies for NWP (HRRR, RRFS) support:

```console
pip install 'cosecha[nwp]'
```

**Note:** Cosecha depends on the [ecCodes](https://confluence.ecmwf.int/display/ECC) C
library for reading GRIB2 data (used by MRMS). When installing with pip, you must have
ecCodes available on your system. The easiest cross-platform approach is to install it
via conda-forge:

```console
conda install -c conda-forge eccodes
pip install cosecha
```

Or use [pixi](https://pixi.sh) which handles this automatically:

```console
pixi add cosecha
```

## Quick Start

```python
from cosecha import USGSNWISReaper

# Fetch USGS streamflow data
reaper = USGSNWISReaper(
    site_ids=["01650000"],
    start_date="2026-01-01",
    end_date="2026-01-31",
    parameter_code="00060",
)

# Execute
data = reaper.reap()

# Write to Parquet
path = reaper.sow_to_parquet(file_path="./data/streamflow.pq")
```

Gridded reapers work the same way. For example, the latest WPC 6-hour QPF for the first
72 hours over a bounding box:

```python
from cosecha import WPCQPFReaper

reaper = WPCQPFReaper(
    init_time="latest",
    forecast_hours=range(6, 73, 6),
    transformations={
        "spatial_subset": {"lat_bounds": (29, 31), "lon_bounds": (-96, -94)},
    },
)
qpf = reaper.reap()
```

## Documentation

Full documentation at <https://dewberry.github.io/cosecha/>

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

MIT License. See [LICENSE](LICENSE.md) for details.
