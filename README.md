# Cosecha

Tools for harvesting earth observation data for use in flood forecasting.

Cosecha provides a flexible pipeline for collecting geospatial data from multiple
sources and writing to various formats with optional transformations.

## Features

- Time-series data collection (NWIS, USGS)
- Gridded data support (HRRR, HRRRv3)
- Multiple output formats: Parquet, NetCDF, Zarr, Iceberg
- Data transformations: unit conversion, spatial subsetting, variable selection
- Full test coverage with zero external service dependencies
- Cross-platform support

## Installation

```console
pip install cosecha
```

Or with pixi:

```console
pixi add cosecha
```

## Quick Start

```python
from cosecha import USGSStreamflowReaper, ParquetSower

# Fetch USGS streamflow data
reaper = USGSStreamflowReaper(site_ids=["01650000"], start_date="2026-01-01", end_date="2026-01-31")

# Execute
data = reaper.reap()

# Write to Parquet
path = data.sow_to_parquet(output_dir="./data")

```

## Documentation

Full documentation at https://dewberry.github.io/cosecha/

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

MIT License. See [LICENSE](LICENSE.md) for details.
