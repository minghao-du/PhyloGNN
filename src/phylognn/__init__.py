"""
PhyloGNN public package surface.

The root package intentionally exposes the primary workflow entry points for
feature engineering, graph conversion, training, and model use. Richer but
still curated APIs remain available in `phylognn.data`, `phylognn.models`, and
`phylognn.training`. Optional tree-loading helpers live under `phylognn.io`.
"""

from importlib import import_module
from typing import Any, Dict, Tuple

__version__ = "0.1.0"

_EXPORT_MAP: Dict[str, Tuple[str, str]] = {
    "attach_node_targets": ("phylognn.data", "attach_node_targets"),
    "TreeFeatureEngineer": ("phylognn.data", "TreeFeatureEngineer"),
    "TreeToGraphConverter": ("phylognn.data", "TreeToGraphConverter"),
    "TrainingConfig": ("phylognn.training", "TrainingConfig"),
    "Trainer": ("phylognn.training", "Trainer"),
    "GATBiLSTMNet": ("phylognn.models", "GATBiLSTMNet"),
    "GATNodeRegressor": ("phylognn.models", "GATNodeRegressor"),
    "MaskedAttentionPhyloRegressor": ("phylognn.models", "MaskedAttentionPhyloRegressor"),
    "TemporalBiLSTMEncoder": ("phylognn.models", "TemporalBiLSTMEncoder"),
    "RegionAssociationResult": ("phylognn.association", "RegionAssociationResult"),
    "build_leaf_laplacian": ("phylognn.association", "build_leaf_laplacian"),
    "evaluate_region_association": ("phylognn.association", "evaluate_region_association"),
}

__all__ = [
    "attach_node_targets",
    "TreeFeatureEngineer",
    "TreeToGraphConverter",
    "TrainingConfig",
    "Trainer",
    "GATBiLSTMNet",
    "GATNodeRegressor",
    "MaskedAttentionPhyloRegressor",
    "TemporalBiLSTMEncoder",
    "RegionAssociationResult",
    "build_leaf_laplacian",
    "evaluate_region_association",
    "__version__",
]


def __getattr__(name: str) -> Any:
    """Lazily resolve curated public exports."""
    if name == "__version__":
        return __version__
    if name in _EXPORT_MAP:
        module_name, attr_name = _EXPORT_MAP[name]
        module = import_module(module_name)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> Any:
    """Return the standard module namespace plus curated exports."""
    return sorted(set(globals()) | set(__all__))
