"""Regression tests for scientific data-contract behavior."""

import pytest
import torch

from tests.support import require_modules


require_modules("torch", "torch_geometric", "ete3")

from ete3 import Tree
from torch_geometric.data import Data

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


def test_converter_preserves_original_node_names():
    """Named tree nodes should remain visible in the converted graph."""
    tree = Tree("((A:1,B:1)C:1,D:1)Root:0;", format=1)
    engineer = TreeFeatureEngineer(num_time_bins=4)
    tree = engineer.add_features(tree, origin_time=3.0, rescale=False)
    converter = TreeToGraphConverter(
        feature_names=engineer.feature_names,
        preserve_node_names=True,
    )

    data = converter.convert(tree)

    assert data.node_names == ["Root", "C", "A", "B", "D"]


def test_converter_rejects_out_of_range_time_bins_for_virtual_nodes():
    """Configured virtual-node bins must cover every original-node time bin."""
    tree = Tree("((A:1,B:1)C:1,D:1)Root:0;", format=1)
    engineer = TreeFeatureEngineer(num_time_bins=4)
    tree = engineer.add_features(tree, origin_time=3.0, rescale=False)
    converter = TreeToGraphConverter(
        feature_names=engineer.feature_names,
        add_virtual_nodes=True,
        num_time_bins=2,
    )

    with pytest.raises(ValueError, match=r"outside configured range \[0, 1\]"):
        converter.convert(tree)


def test_converter_asserts_integer_like_time_bins_for_virtual_nodes():
    """Virtual-node construction should reject silently truncated time bins."""
    converter = TreeToGraphConverter(
        feature_names=("time_bin",),
        add_virtual_nodes=True,
        num_time_bins=3,
    )
    data = Data(
        x=torch.tensor([[0.0], [1.7], [2.0]], dtype=torch.float32),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_type=torch.empty((0,), dtype=torch.long),
        original_num_nodes=3,
    )

    with pytest.raises(AssertionError, match="integer-like"):
        converter._add_virtual_nodes(data)
