# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Add reaping class and exceptions
- Add nwis and nwp reapers
- Add sowing base class and exceptions
- Add sowing classes
- Add cosecha top level code
- Add dataretrieval for nwis reaper
- Generalize HRRReaper to accept other herbie models
- Add MRMS reaper
- Update spatial subset to work on 1d and 2d lat/lon coords
- Add ability to cache mrms files

### Changed

- Consolidate apply_transormations methods into utils script
- Remove placeholder scripts
- Move transformations from sowers to reapers
- Refactor USGS classes into one class
- Major refactor: Create base reaper classes with sowers. Remove HarvestedData
- Replace logging_config with logging module
- Consolidate exception handling with wrap_errors context manager
- Consolidate base reaper classes and utilities
- Replace rich logging with stdlib StreamHandler
- Fix ruff lint warnings in tests

### Fix

- Update pyproj dependencies

### Fixed

- Add search for matching vars in nwp
- Updated dependencies
- Fix icechunk sower so it actually writes to an icechunk store
- Remove network tests
- Remove network tests
- Add eccodes as dependencies
- Move unneeded optional dependencies
- Fix spelling
- Remove duplicate dependencies, ignore 'docs' folder in linter
- Use custom logger
- Allow time in format YYYY-MM-DDTHH:MMZ for usgs
- Add transformation to class init, fix docs, remove stat code
- Remove unused args
- Fix file path naming
- Use waterdata API for USGS data instead of NWIS
- Remove unnecessary mkdir
- Improve reapers with consistent parsing and public exports

### New Contributors

- @slawler made their first contribution


