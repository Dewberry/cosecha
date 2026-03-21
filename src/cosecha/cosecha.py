"""Core module for cosecha."""

from __future__ import annotations


def hello(name: str = "world") -> str:
    """Return a greeting string.

    Parameters
    ----------
    name : str, optional
        The name to greet, by default "world".

    Returns
    -------
    str
        A greeting string.

    Examples
    --------
    >>> hello()
    'Hello, world!'
    >>> hello("Alice")
    'Hello, Alice!'
    """
    return f"Hello, {name}!"
