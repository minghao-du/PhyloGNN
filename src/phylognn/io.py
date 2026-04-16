"""
Explicit tree I/O entry points for PhyloGNN.

These helpers live outside the default package and data import surfaces so
users who do not need optional tree-loading functionality are not coupled to
the extra dependency chain at import time.
"""

from importlib import import_module
from typing import Any, Dict, Tuple


_EXPORT_MAP: Dict[str, Tuple[str, str]] = {
    "TreeReadConfig": ("phylognn.data.tree_io", "TreeReadConfig"),
    "read_tree_as_ete3": ("phylognn.data.tree_io", "read_tree_as_ete3"),
    "read_tree_with_dendropy": ("phylognn.data.tree_io", "read_tree_with_dendropy"),
    "dendropy_tree_to_ete3": ("phylognn.data.tree_io", "dendropy_tree_to_ete3"),
}

__all__ = list(_EXPORT_MAP)


def __getattr__(name: str) -> Any:
    """Lazily resolve optional tree I/O exports."""
    if name in _EXPORT_MAP:
        module_name, attr_name = _EXPORT_MAP[name]
        module = import_module(module_name)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> Any:
    """Return the standard module namespace plus curated exports."""
    return sorted(set(globals()) | set(__all__))
