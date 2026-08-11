"""Tests for curated package export surfaces that do not require runtime deps."""

import importlib

import pytest

torch = pytest.importorskip("torch")


def test_root_package_exposes_leaf_regression_public_names():
    """The root package should publish only the supported stable surface."""
    phylognn = importlib.import_module("phylognn")

    assert phylognn.__version__ == "0.1.0"
    assert phylognn.__all__ == [
        "attach_node_targets",
        "TreeFeatureEngineer",
        "TreeToGraphConverter",
        "TrainingConfig",
        "Trainer",
        "GATBiLSTMNet",
        "GATNodeRegressor",
        "MaskedAttentionPhyloRegressor",
        "TemporalBiLSTMEncoder",
        "LeafRegressionData",
        "LeafRegressionConfig",
        "LeafFitResult",
        "LeafCrossValidationResult",
        "LeafRegressionResult",
        "prepare_leaf_regression",
        "fit_leaf_regression",
        "cross_validate_leaf_regression",
        "run_leaf_regression",
        "__version__",
    ]
    assert "read_tree_as_ete3" not in phylognn.__all__


def test_data_subpackage_keeps_tree_io_off_default_surface():
    """The data package should expose only the core preprocessing pipeline."""
    data = importlib.import_module("phylognn.data")

    assert data.__all__ == ["attach_node_targets", "TreeFeatureEngineer", "TreeToGraphConverter"]
    assert "read_tree_as_ete3" not in data.__all__


def test_models_subpackage_hides_low_level_layers():
    """Low-level model layers should not be package-level exports."""
    models = importlib.import_module("phylognn.models")

    assert models.__all__ == [
        "BasePhyloGNN",
        "BaseGATNet",
        "GATBiLSTMNet",
        "GATNodeRegressor",
        "MaskedAttentionPhyloRegressor",
        "SparseQueryPhyloRegressor",
        "TemporalBiLSTMEncoder",
    ]
    assert models.TemporalBiLSTMEncoder.__name__ == "TemporalBiLSTMEncoder"
    assert models.MaskedAttentionPhyloRegressor.__name__ == "MaskedAttentionPhyloRegressor"
    assert models.SparseQueryPhyloRegressor.__name__ == "SparseQueryPhyloRegressor"
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


def test_root_package_lazily_exports_leaf_regression_api():
    """Every Leaf Regression contract resolves through both public facades."""
    import phylognn
    import phylognn.leaf_regression

    for name in (
        "LeafRegressionData",
        "LeafRegressionConfig",
        "LeafFitResult",
        "LeafCrossValidationResult",
        "LeafRegressionResult",
        "prepare_leaf_regression",
        "fit_leaf_regression",
        "cross_validate_leaf_regression",
        "run_leaf_regression",
    ):
        assert name in phylognn.__all__
        assert name in dir(phylognn)
        assert getattr(phylognn, name) is getattr(phylognn.leaf_regression, name)

    assert not any("reg" + "ion" in name.lower() for name in phylognn.__all__)
    assert not any("association" in name.lower() for name in phylognn.__all__)
    assert not any("factory" in name.lower() for name in phylognn.__all__)


def test_target_attachment_is_a_lazy_public_export():
    """Target attachment should work from both curated public facades."""
    from torch_geometric.data import Data

    import phylognn
    import phylognn.data

    graph = Data(x=torch.ones((2, 1)), node_names=["A", "B"])
    result = phylognn.attach_node_targets(graph, {"B": 2.0, "A": 1.0})

    assert phylognn.attach_node_targets is phylognn.data.attach_node_targets
    assert result.y.tolist() == [1.0, 2.0]
    assert "attach_node_targets" in dir(phylognn)
    assert "attach_node_targets" in dir(phylognn.data)


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


def test_supported_loss_names_is_exported_from_training_but_not_the_root_package():
    """supported_loss_names belongs to phylognn.training, not the curated root surface."""
    import phylognn
    import phylognn.training

    assert "supported_loss_names" in phylognn.training.__all__
    assert "supported_loss_names" in dir(phylognn.training)
    assert callable(phylognn.training.supported_loss_names)
    assert "supported_loss_names" not in phylognn.__all__

    try:
        getattr(phylognn, "supported_loss_names")
    except AttributeError:
        pass
    else:
        raise AssertionError("Root package unexpectedly exposed supported_loss_names.")
