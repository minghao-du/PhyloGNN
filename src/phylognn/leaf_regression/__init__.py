"""Curated public API for leaf-level regression on phylogenetic trees."""

from typing import TYPE_CHECKING

from .data import LeafRegressionData, prepare_leaf_regression
from .fitting import LeafFitResult, LeafRegressionConfig, fit_leaf_regression
from .validation import (
    LeafCrossValidationResult,
    LeafRegressionResult,
    cross_validate_leaf_regression,
    run_leaf_regression,
)

if TYPE_CHECKING:
    from .tracking import _LeafExperimentCoordinator, _LeafTrackingStage  # noqa: F401

__all__ = [
    "LeafRegressionData",
    "LeafRegressionConfig",
    "LeafFitResult",
    "LeafCrossValidationResult",
    "LeafRegressionResult",
    "prepare_leaf_regression",
    "fit_leaf_regression",
    "cross_validate_leaf_regression",
    "run_leaf_regression",
]
