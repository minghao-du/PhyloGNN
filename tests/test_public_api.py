"""Tests for curated package export surfaces that do not require runtime deps."""

import importlib


def test_root_package_exposes_curated_public_names():
    """The root package should publish only the supported stable surface."""
    phylognn = importlib.import_module("phylognn")

    assert phylognn.__version__ == "0.1.0"
    assert phylognn.__all__ == [
        "TreeFeatureEngineer",
        "TreeToGraphConverter",
        "TrainingConfig",
        "Trainer",
        "GATBiLSTMNet",
        "GATNodeRegressor",
        "TemporalBiLSTMEncoder",
        "__version__",
    ]
    assert "read_tree_as_ete3" not in phylognn.__all__


def test_data_subpackage_keeps_tree_io_off_default_surface():
    """The data package should expose only the core preprocessing pipeline."""
    data = importlib.import_module("phylognn.data")

    assert data.__all__ == ["TreeFeatureEngineer", "TreeToGraphConverter"]
    assert "read_tree_as_ete3" not in data.__all__


def test_models_subpackage_hides_low_level_layers():
    """Low-level model layers should not be package-level exports."""
    models = importlib.import_module("phylognn.models")

    assert models.__all__ == [
        "BasePhyloGNN",
        "BaseGATNet",
        "GATBiLSTMNet",
        "GATNodeRegressor",
        "TemporalBiLSTMEncoder",
    ]
    assert models.TemporalBiLSTMEncoder.__name__ == "TemporalBiLSTMEncoder"
    for hidden_name in {
        "GATBlock",
        "ResidualGATStack",
        "PositionalEncoding",
        "MLPHead",
    }:
        assert hidden_name not in models.__all__


def test_io_module_defines_explicit_optional_tree_io_boundary():
    """Tree I/O should be available from a dedicated module boundary."""
    io = importlib.import_module("phylognn.io")

    assert "read_tree_as_ete3" in io.__all__
    assert "TreeReadConfig" in io.__all__


def test_root_package_dir_matches_curated_surface():
    """Directory listings should advertise the same curated root package API."""
    phylognn = importlib.import_module("phylognn")

    for export_name in phylognn.__all__:
        assert export_name in dir(phylognn)

    assert phylognn.TemporalBiLSTMEncoder.__name__ == "TemporalBiLSTMEncoder"


def test_root_package_rejects_hidden_optional_tree_io_names():
    """Optional helpers must not leak into the root package namespace."""
    phylognn = importlib.import_module("phylognn")

    try:
        getattr(phylognn, "read_tree_as_ete3")
    except AttributeError:
        pass
    else:
        raise AssertionError("Root package unexpectedly exposed optional tree I/O helper.")


def test_public_runtime_facades_import_from_default_dependency_profile():
    """Default dependency metadata should be sufficient for public runtime facades."""
    for module_name in ("phylognn", "phylognn.data", "phylognn.models", "phylognn.training"):
        module = importlib.import_module(module_name)

        assert module.__all__
