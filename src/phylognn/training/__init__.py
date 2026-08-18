"""
Curated training-facing API for PhyloGNN.

This package exposes split-aware dataset types, trainer utilities, and
configuration helpers intended for end-user workflows.
"""

from importlib import import_module
from typing import Any, Dict, Tuple

_EXPORT_MAP: Dict[str, Tuple[str, str]] = {
    "ConfiguredTrainingSetup": ("phylognn.training.config", "ConfiguredTrainingSetup"),
    "DatasetSplit": ("phylognn.training.dataset", "DatasetSplit"),
    "SplitDatasetView": ("phylognn.training.dataset", "SplitDatasetView"),
    "SplitPhyloDataset": ("phylognn.training.dataset", "SplitPhyloDataset"),
    "SplitPhyloDiskDataset": ("phylognn.training.dataset", "SplitPhyloDiskDataset"),
    "Trainer": ("phylognn.training.trainer", "Trainer"),
    "TrainingConfigError": ("phylognn.training.config", "TrainingConfigError"),
    "TrainingConfig": ("phylognn.training.trainer", "TrainingConfig"),
    "TrackingConfig": ("phylognn.training.tracking", "TrackingConfig"),
    "TrackingError": ("phylognn.training.tracking", "TrackingError"),
    "TrackingRunInfo": ("phylognn.training.tracking", "TrackingRunInfo"),
    "WandbTracker": ("phylognn.training.tracking", "WandbTracker"),
    "create_default_trainer": ("phylognn.training.trainer", "create_default_trainer"),
    "create_trainer_from_config": ("phylognn.training.config", "create_trainer_from_config"),
    "load_training_config": ("phylognn.training.config", "load_training_config"),
    "supported_loss_names": ("phylognn.training.losses", "supported_loss_names"),
    "PGLSLoss": ("phylognn.training.losses", "PGLSLoss"),
}

__all__ = [
    "ConfiguredTrainingSetup",
    "DatasetSplit",
    "SplitDatasetView",
    "SplitPhyloDataset",
    "SplitPhyloDiskDataset",
    "Trainer",
    "TrainingConfigError",
    "TrainingConfig",
    "TrackingConfig",
    "TrackingError",
    "TrackingRunInfo",
    "WandbTracker",
    "create_default_trainer",
    "create_trainer_from_config",
    "load_training_config",
    "supported_loss_names",
    "PGLSLoss",
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
