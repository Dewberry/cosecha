"""Cosecha: Tools for harvesting earth observation data for use in flood forecasting."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from cosecha.cosecha import hello

try:
    __version__ = version("cosecha")
except PackageNotFoundError:
    __version__ = "999"

__all__ = [
    "__version__",
    "hello",
]
