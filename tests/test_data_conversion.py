"""Focused tests for TreeToGraphConverter public contracts."""

import pytest


pytest.importorskip("ete3")
pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from ete3 import Tree

from phylognn.data import TreeFeatureEngineer, TreeToGraphConverter


def test_output_feature_names_use_canonical_virtual_node_name():
    """The appended virtual-node flag should use one canonical name."""
    converter = TreeToGraphConverter(
        feature_names=("node_time", "time_bin"),
        append_is_virtual_feature=True,
    )

    assert converter.output_feature_names[-1] == "is_virtual_node"


def test_convert_appends_virtual_node_feature_column_consistently():
    """Converted graphs should expose the canonical virtual-node feature name."""
    tree = Tree("((A:1,B:1)C:1,D:1)E:0;", format=1)
    engineer = TreeFeatureEngineer(num_time_bins=5)
    tree = engineer.add_features(tree, origin_time=2.0, rescale=False)

    converter = TreeToGraphConverter(
        feature_names=engineer.feature_names,
        add_virtual_nodes=True,
        num_time_bins=engineer.num_time_bins,
        append_is_virtual_feature=True,
    )
    data = converter.convert(tree)

    assert converter.output_feature_names[-1] == "is_virtual_node"
    assert data.x.shape[1] == len(converter.output_feature_names)
    assert data.virtual_node_mask.any()
