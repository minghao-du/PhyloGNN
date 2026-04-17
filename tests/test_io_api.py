"""Tests for the optional tree I/O facade contract."""

import importlib

import pytest


def test_io_module_exports_only_optional_tree_io_surface():
    """The optional I/O boundary should stay explicit and curated."""
    io = importlib.import_module("phylognn.io")

    assert io.__all__ == [
        "TreeReadConfig",
        "read_tree_as_ete3",
        "read_tree_with_dendropy",
        "dendropy_tree_to_ete3",
    ]


def test_io_module_dir_includes_curated_names():
    """The module directory should expose all curated optional names."""
    io = importlib.import_module("phylognn.io")

    visible_names = dir(io)
    for name in io.__all__:
        assert name in visible_names


def test_io_module_rejects_unknown_attributes():
    """Unknown optional I/O attributes should fail loudly."""
    io = importlib.import_module("phylognn.io")

    with pytest.raises(AttributeError):
        getattr(io, "not_a_real_tree_io_symbol")


def test_tree_io_helpers_do_not_leak_into_default_surfaces():
    """Optional tree I/O remains outside the root and data default facades."""
    root = importlib.import_module("phylognn")
    data = importlib.import_module("phylognn.data")

    assert "read_tree_as_ete3" not in root.__all__
    assert "read_tree_as_ete3" not in data.__all__
