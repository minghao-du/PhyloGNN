"""Shared helpers for contract-style tests in this repository."""

from pathlib import Path
from typing import Iterable, List
import importlib

import pytest


def require_modules(*module_names: str):
    """Import modules or skip the calling test if any dependency is missing."""
    imported = []
    for module_name in module_names:
        imported.append(pytest.importorskip(module_name))
    return imported if len(imported) > 1 else imported[0]


def scientific_stack():
    """Return the common scientific stack needed by data/model/training tests."""
    return require_modules("torch", "torch_geometric", "ete3")


def optional_tree_io_stack():
    """Return the optional tree I/O stack when available."""
    return require_modules("ete3", "dendropy")


def import_from_path(module_name: str):
    """Import a module by dotted path."""
    return importlib.import_module(module_name)


def project_root() -> Path:
    """Return the repository root for path-based assertions."""
    return Path(__file__).resolve().parents[1]


def sorted_public_names(names: Iterable[str]) -> List[str]:
    """Normalize a name collection for stable equality checks."""
    return sorted(names)
