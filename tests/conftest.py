"""Pytest configuration and fixtures."""

from __future__ import annotations

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "requires_netcdf: mark test as requiring netCDF4 system libraries"
    )
    config.addinivalue_line(
        "markers", "requires_herbie: mark test as requiring herbie optional dependency"
    )
    config.addinivalue_line(
        "markers", "requires_iceberg: mark test as requiring Iceberg catalog configuration"
    )


netcdf_available = False
try:
    # netCDF4 preferred for performance
    import netCDF4  # noqa: F401

    netcdf_available = True
except (ImportError, OSError):
    try:
        import h5netcdf  # noqa: F401

        netcdf_available = True
    except Exception:
        netcdf_available = False

herbie_available = False
try:
    import herbie  # noqa: F401

    herbie_available = True
except Exception:
    herbie_available = False

iceberg_available = False
try:
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from pyiceberg.catalog.sql import SqlCatalog

    with TemporaryDirectory() as tmpdir:
        metadata_dir = Path(tmpdir) / ".iceberg"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        with SqlCatalog(
            name="test", uri=f"sqlite:///{metadata_dir / 'iceberg.db'}", warehouse=tmpdir
        ):
            iceberg_available = True
except (ImportError, Exception):
    iceberg_available = False


def pytest_collection_modifyitems(config, items):
    """Skip tests that require unavailable dependencies."""
    for item in items:
        if "requires_netcdf" in item.keywords and not netcdf_available:
            item.add_marker(pytest.mark.skip(reason="netCDF4 system libraries not available"))

        if "requires_herbie" in item.keywords and not herbie_available:
            item.add_marker(pytest.mark.skip(reason="Herbie optional dependency not installed"))

        if "requires_iceberg" in item.keywords and not iceberg_available:
            item.add_marker(pytest.mark.skip(reason="Iceberg catalog URI not configured"))
