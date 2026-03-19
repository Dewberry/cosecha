"""Tests for package_name."""

from __future__ import annotations

import pytest

import package_name


def test_hello_default() -> None:
    assert package_name.hello() == "Hello, world!"


def test_hello_custom() -> None:
    assert package_name.hello("Alice") == "Hello, Alice!"


@pytest.mark.network
def test_network_example() -> None:
    # Example network test - replace with real network tests
    pass
