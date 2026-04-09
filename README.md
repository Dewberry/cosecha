# Cosecha

Tools for harvesting earth observation data for use in flood forecasting.

Cosecha provides a flexible pipeline for collecting geospatial data from multiple
sources and writing to various formats with optional transformations.

## Features

- Time-series data collection (USGS NWIS streamflow, stage, precipitation)
- Gridded data support (HRRR, RRFS, RTMA via herbie; MRMS via S3)
- Multiple output formats: Parquet, NetCDF, Zarr, Iceberg, IceChunk
- Data transformations: unit conversion, spatial subsetting, variable selection/rename
- Cross-platform support (ecCodes C library required for GRIB2/MRMS)

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
    site_ids=["01650000"], start_date="2026-01-01", end_date="2026-01-31", parameter_code="00060"
)

# Execute
data = reaper.reap()

# Write to Parquet
path = reaper.sow_to_parquet(file_path="./data/streamflow.pq")
```

## Documentation

Full documentation at <https://dewberry.github.io/cosecha/>

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

MIT License. See [LICENSE](LICENSE.md) for details.
