"""Tests for the curated model facade contract."""

import importlib

import pytest


def test_models_package_exports_supported_model_surface():
    """The models package should expose only the supported model-level API."""
    models = importlib.import_module("phylognn.models")

    assert models.__all__ == [
        "BasePhyloGNN",
        "BaseGATNet",
        "GATBiLSTMNet",
        "GATNodeRegressor",
        "MaskedAttentionPhyloRegressor",
        "OneHotPhyloRegressor",
        "SparseQueryPhyloRegressor",
        "TemporalBiLSTMEncoder",
    ]


def test_models_package_dir_contains_curated_names():
    """The package directory should include every curated public model symbol."""
    models = importlib.import_module("phylognn.models")

    for name in models.__all__:
        assert name in dir(models)


def test_models_package_rejects_low_level_helpers_from_facade():
    """Low-level implementation helpers stay off the package facade."""
    models = importlib.import_module("phylognn.models")

    assert models.TemporalBiLSTMEncoder.__name__ == "TemporalBiLSTMEncoder"
    assert models.MaskedAttentionPhyloRegressor.__name__ == "MaskedAttentionPhyloRegressor"
    assert models.OneHotPhyloRegressor.__name__ == "OneHotPhyloRegressor"
    assert models.SparseQueryPhyloRegressor.__name__ == "SparseQueryPhyloRegressor"

    for hidden_name in {"GATBlock", "ResidualGATBlock", "ResidualGATStack", "MLPHead"}:
        assert hidden_name not in models.__all__

    with pytest.raises(AttributeError):
        getattr(models, "GATBlock")
