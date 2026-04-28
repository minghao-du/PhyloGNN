"""
Curated data-facing API for PhyloGNN.

This package exposes the primary preprocessing pipeline:

1. `TreeFeatureEngineer`
2. `TreeToGraphConverter`

Optional tree-loading helpers are intentionally separated under `phylognn.io`
so users of the core data pipeline are not coupled to optional file-parsing
dependencies at import time.
"""

from importlib import import_module
from typing import Any, Dict, Tuple

_EXPORT_MAP: Dict[str, Tuple[str, str]] = {
    "TreeFeatureEngineer": ("phylognn.data.feature_engineer", "TreeFeatureEngineer"),
    "TreeToGraphConverter": ("phylognn.data.converter", "TreeToGraphConverter"),
}

__all__ = ["TreeFeatureEngineer", "TreeToGraphConverter"]


def __getattr__(name: str) -> Any:
    """Lazily resolve curated data exports."""
    if name in _EXPORT_MAP:
        module_name, attr_name = _EXPORT_MAP[name]
        module = import_module(module_name)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> Any:
    """Return the standard module namespace plus curated exports."""
    return sorted(set(globals()) | set(__all__))
