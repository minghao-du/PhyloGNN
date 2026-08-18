"""
Curated model-facing API for PhyloGNN.

Package-level exports are limited to supported base classes and end-user model
types. Low-level layers remain available from explicit module paths.
"""

from importlib import import_module
from typing import Any, Dict, Tuple

_EXPORT_MAP: Dict[str, Tuple[str, str]] = {
    "BasePhyloGNN": ("phylognn.models.base", "BasePhyloGNN"),
    "BaseGATNet": ("phylognn.models.base", "BaseGATNet"),
    "GATBiLSTMNet": ("phylognn.models.gat_lstm", "GATBiLSTMNet"),
    "GATNodeRegressor": ("phylognn.models.gat_node", "GATNodeRegressor"),
    "MaskedAttentionPhyloRegressor": (
        "phylognn.models.masked_attention",
        "MaskedAttentionPhyloRegressor",
    ),
    "OneHotPhyloRegressor": (
        "phylognn.models.one_hot_phylo",
        "OneHotPhyloRegressor",
    ),
    "SparseQueryPhyloRegressor": (
        "phylognn.models.sparse_query",
        "SparseQueryPhyloRegressor",
    ),
    "TemporalBiLSTMEncoder": ("phylognn.models.layers", "TemporalBiLSTMEncoder"),
    "PGLSRegressionHead": ("phylognn.models.pgls", "PGLSRegressionHead"),
}

__all__ = [
    "BasePhyloGNN",
    "BaseGATNet",
    "GATBiLSTMNet",
    "GATNodeRegressor",
    "MaskedAttentionPhyloRegressor",
    "OneHotPhyloRegressor",
    "SparseQueryPhyloRegressor",
    "TemporalBiLSTMEncoder",
    "PGLSRegressionHead",
]


def __getattr__(name: str) -> Any:
    """Lazily resolve curated model exports."""
    if name in _EXPORT_MAP:
        module_name, attr_name = _EXPORT_MAP[name]
        module = import_module(module_name)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> Any:
    """Return the standard module namespace plus curated exports."""
    return sorted(set(globals()) | set(__all__))
