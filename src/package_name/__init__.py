"""PackageName: A short description of the project."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from package_name.package_name import hello

try:
    __version__ = version("package_name")
except PackageNotFoundError:
    __version__ = "999"

__all__ = [
    "__version__",
    "hello",
]
