"""
Curated training-facing API for PhyloGNN.

This package exposes split-aware dataset types, trainer utilities, and public
metric helpers intended for end-user workflows.
"""

from importlib import import_module
from typing import Any, Dict, Tuple


_EXPORT_MAP: Dict[str, Tuple[str, str]] = {
    "DatasetSplit": ("phylognn.training.dataset", "DatasetSplit"),
    "SplitDatasetView": ("phylognn.training.dataset", "SplitDatasetView"),
    "SplitPhyloDataset": ("phylognn.training.dataset", "SplitPhyloDataset"),
    "SplitPhyloDiskDataset": ("phylognn.training.dataset", "SplitPhyloDiskDataset"),
    "Trainer": ("phylognn.training.trainer", "Trainer"),
    "TrainingConfig": ("phylognn.training.trainer", "TrainingConfig"),
    "create_default_trainer": ("phylognn.training.trainer", "create_default_trainer"),
    "mse_metric": ("phylognn.training.metrics", "mse_metric"),
    "mae_metric": ("phylognn.training.metrics", "mae_metric"),
    "r2_metric": ("phylognn.training.metrics", "r2_metric"),
    "rmse_metric": ("phylognn.training.metrics", "rmse_metric"),
    "relative_error_metric": ("phylognn.training.metrics", "relative_error_metric"),
}

__all__ = [
    "DatasetSplit",
    "SplitDatasetView",
    "SplitPhyloDataset",
    "SplitPhyloDiskDataset",
    "Trainer",
    "TrainingConfig",
    "create_default_trainer",
    "mse_metric",
    "mae_metric",
    "r2_metric",
    "rmse_metric",
    "relative_error_metric",
]


def __getattr__(name: str) -> Any:
    """Lazily resolve curated training exports."""
    if name in _EXPORT_MAP:
        module_name, attr_name = _EXPORT_MAP[name]
        module = import_module(module_name)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> Any:
    """Return the standard module namespace plus curated exports."""
    return sorted(set(globals()) | set(__all__))
