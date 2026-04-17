"""Tests for the utility facade and helper contracts."""

import importlib


class _FakeTree:
    """Minimal traverse-compatible stand-in for utility tests."""

    def __init__(self, nodes):
        self._nodes = nodes

    def traverse(self):
        return iter(self._nodes)


class _Node:
    """Simple object with optional metadata attributes."""

    def __init__(self, **attrs):
        for key, value in attrs.items():
            setattr(self, key, value)


def test_utils_package_exports_only_supported_helper():
    """The utils facade should stay intentionally minimal."""
    utils = importlib.import_module("phylognn.utils")

    assert utils.__all__ == ["get_max_meta_time"]
    assert "get_max_meta_time" in dir(utils)


def test_get_max_meta_time_returns_highest_numeric_metadata():
    """The helper should compute the maximum metadata value deterministically."""
    utils = importlib.import_module("phylognn.utils")
    tree = _FakeTree([_Node(meta_time="2.5"), _Node(meta_time=7), _Node(other=3)])

    assert utils.get_max_meta_time(tree) == 7.0


def test_get_max_meta_time_returns_none_when_metadata_is_missing():
    """Trees without metadata should produce a clear null result."""
    utils = importlib.import_module("phylognn.utils")
    tree = _FakeTree([_Node(other=1), _Node(other=2)])

    assert utils.get_max_meta_time(tree) is None
