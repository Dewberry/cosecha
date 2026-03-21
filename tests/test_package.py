"""Tests for cosecha."""

from __future__ import annotations

import pytest

import cosecha


def test_hello_default() -> None:
    assert cosecha.hello() == "Hello, world!"


def test_hello_custom() -> None:
    assert cosecha.hello("Alice") == "Hello, Alice!"


@pytest.mark.network
def test_network_example() -> None:
    # Example network test - replace with real network tests
    pass
