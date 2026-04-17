"""Regression tests for scientific data-contract behavior."""

import pytest

from tests.support import require_modules


require_modules("torch", "torch_geometric", "ete3")

from ete3 import Tree

from phylognn.data import TreeFeatureEngineer, TreeToGraphConverter


def test_feature_engineer_rejects_duplicate_requested_features():
    """Feature requests should use a unique stable column order."""
    engineer = TreeFeatureEngineer()
    tree = Tree("(A:1,B:1)Root;", format=1)

    with pytest.raises(ValueError, match="must not contain duplicates"):
        engineer.add_features(
            tree,
            origin_time=2.0,
            feature_names=("node_time", "node_time"),
            rescale=False,
        )


def test_converter_rejects_duplicate_feature_names():
    """The converter should fail early on ambiguous feature-column definitions."""
    with pytest.raises(ValueError, match="must not contain duplicates"):
        TreeToGraphConverter(feature_names=("time_bin", "time_bin"))


def test_converter_preserves_deterministic_node_names_with_virtual_nodes():
    """Repeated conversions should preserve node-name ordering and virtual suffixes."""
    tree = Tree("((A:1,B:1)C:1,D:1)E:0;", format=1)
    engineer = TreeFeatureEngineer(num_time_bins=4)
    tree = engineer.add_features(tree, origin_time=3.0, rescale=False)
    converter = TreeToGraphConverter(
        feature_names=engineer.feature_names,
        add_virtual_nodes=True,
        num_time_bins=engineer.num_time_bins,
        append_is_virtual_feature=True,
        preserve_node_names=True,
    )

    first = converter.convert(tree)
    second = converter.convert(tree)

    assert first.node_names == second.node_names
    assert first.node_names[-1] == "__virtual_time_bin_3__"
