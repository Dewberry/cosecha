# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-04-23

### Added

- Add subhourly time handling
- Add "latest" time option when reaping data
- Allow selection of multiple NWP variables
- Add "latest" option when reaping model data

### Fixed

- Consolidate time args into one 'dates' arg
## [0.1.0] - 2026-04-08

### Added

- Add reaping class and exceptions by @slawler
- Add nwis and nwp reapers by @slawler
- Add sowing base class and exceptions by @slawler
- Add sowing classes by @slawler
- Add cosecha top level code by @slawler
- Add dataretrieval for nwis reaper by @slawler
- Generalize HRRReaper to accept other herbie models by @sray014
- Add MRMS reaper by @sray014
- Update spatial subset to work on 1d and 2d lat/lon coords by @sray014
- Add ability to cache mrms files by @sray014
- Add S3 support for parquet, zarr, and icechunk writers

### Changed

- Consolidate apply_transormations methods into utils script by @sray014
- Remove placeholder scripts by @sray014
- Move transformations from sowers to reapers by @sray014
- Refactor USGS classes into one class by @sray014
- Major refactor: Create base reaper classes with sowers. Remove HarvestedData by @sray014
- Replace logging_config with logging module
- Consolidate exception handling with wrap_errors context manager
- Consolidate base reaper classes and utilities
- Replace rich logging with stdlib StreamHandler
- Fix ruff lint warnings in tests
- Remove unused SpatialBoundsError and exception re-exports
- Make utils and logging modules private
- Use TypeVar instead of overload decorator by @sray014

### Fix

- Update pyproj dependencies by @sray014

### Fixed

- Add search for matching vars in nwp by @slawler
- Updated dependencies by @sray014
- Fix icechunk sower so it actually writes to an icechunk store by @sray014
- Remove network tests by @sray014
- Remove network tests by @sray014
- Add eccodes as dependencies by @sray014
- Move unneeded optional dependencies by @sray014
- Fix spelling by @sray014
- Remove duplicate dependencies, ignore 'docs' folder in linter by @sray014
- Use custom logger by @sray014
- Allow time in format YYYY-MM-DDTHH:MMZ for usgs by @sray014
- Add transformation to class init, fix docs, remove stat code by @sray014
- Remove unused args by @sray014
- Fix file path naming by @sray014
- Use waterdata API for USGS data instead of NWIS by @sray014
- Remove unnecessary mkdir by @sray014
- Improve reapers with consistent parsing and public exports
- Correct LICENSE path in mkdocs nav
- Restore LICENSE.md path in mkdocs nav to match docs hook
- Add eccodes to pixi envs and broaden conftest exception handling
- Use FsspecFileIO for Iceberg to support Windows paths
- Use file URI for Iceberg warehouse on Windows
- Use ensure_local_parent when writing to netcdf by @sray014

### New Contributors

- @sray014 made their first contribution in [#1](https://github.com/Dewberry/cosecha/pull/1)
- @ made their first contribution
- @slawler made their first contribution
[0.1.1]: https://github.com/Dewberry/cosecha/compare/v0.1.0...v0.1.1

