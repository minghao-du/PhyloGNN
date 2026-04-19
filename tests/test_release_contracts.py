"""Release-facing smoke tests for public contracts."""

import importlib

import pytest


@pytest.mark.parametrize(
    ("module_name", "expected_names"),
    [
        ("phylognn", {"TreeFeatureEngineer", "TreeToGraphConverter", "TrainingConfig", "Trainer"}),
        ("phylognn.data", {"TreeFeatureEngineer", "TreeToGraphConverter"}),
        ("phylognn.models", {"BasePhyloGNN", "GATBiLSTMNet"}),
        ("phylognn.training", {"Trainer", "TrainingConfig", "rmse_metric"}),
        ("phylognn.io", {"TreeReadConfig", "read_tree_as_ete3"}),
        ("phylognn.utils", {"get_max_meta_time"}),
    ],
)
def test_curated_facades_keep_expected_names(module_name, expected_names):
    """Release validation should confirm curated facades still expose supported names."""
    module = importlib.import_module(module_name)

    assert expected_names.issubset(set(module.__all__))


@pytest.mark.parametrize(
    ("module_name", "hidden_name"),
    [
        ("phylognn", "read_tree_as_ete3"),
        ("phylognn.data", "read_tree_as_ete3"),
        ("phylognn.models", "GATBlock"),
        ("phylognn.training", "all"),
    ],
)
def test_release_contracts_reject_hidden_names(module_name, hidden_name):
    """Release validation should detect leaked internal or optional names."""
    module = importlib.import_module(module_name)

    with pytest.raises(AttributeError):
        getattr(module, hidden_name)
