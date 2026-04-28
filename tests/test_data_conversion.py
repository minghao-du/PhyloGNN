"""Focused tests for TreeToGraphConverter public contracts."""

import pytest

pytest.importorskip("ete3")
pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from ete3 import Tree

from phylognn.data import TreeFeatureEngineer, TreeToGraphConverter


def _feature_index(converter, feature_name):
    return converter.output_feature_names.index(feature_name)


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


def test_converter_rejects_duplicate_feature_names():
    """The converter should fail early on ambiguous feature-column definitions."""
    with pytest.raises(ValueError, match="must not contain duplicates"):
        TreeToGraphConverter(feature_names=("time_bin", "time_bin"))


def test_convert_creates_configured_virtual_nodes_for_rescaled_empty_bins():
    """Configured bins should create virtual nodes even when no original node uses a bin."""
    tree = Tree("((A:1,B:2)C:3,D:4)Root:0;", format=1)
    engineer = TreeFeatureEngineer(
        num_time_bins=5,
        extant_sampling_probability=0.25,
    )
    tree = engineer.add_features(tree, origin_time=5.0, rescale=True)
    converter = TreeToGraphConverter(
        feature_names=engineer.feature_names,
        add_virtual_nodes=True,
        num_time_bins=engineer.num_time_bins,
        append_is_virtual_feature=True,
        preserve_node_names=True,
    )

    data = converter.convert(tree)

    virtual_slice = slice(data.original_num_nodes, None)
    original_time_bins = data.x[: data.original_num_nodes, _feature_index(converter, "time_bin")]
    assert 3.0 not in original_time_bins.tolist()
    assert data.num_time_bins == engineer.num_time_bins
    assert data.virtual_node_mask.sum().item() == engineer.num_time_bins
    assert data.node_names[-5:] == [
        "__virtual_time_bin_0__",
        "__virtual_time_bin_1__",
        "__virtual_time_bin_2__",
        "__virtual_time_bin_3__",
        "__virtual_time_bin_4__",
    ]
    assert data.x[virtual_slice, _feature_index(converter, "time_bin")].tolist() == pytest.approx(
        [0.0, 1.0, 2.0, 3.0, 4.0]
    )
    assert data.x[virtual_slice, _feature_index(converter, "node_time")].tolist() == pytest.approx(
        [0.0] * engineer.num_time_bins
    )
    assert data.x[virtual_slice, _feature_index(converter, "rescale_factor")].tolist() == (
        pytest.approx([0.0] * engineer.num_time_bins)
    )
    assert data.x[
        virtual_slice, _feature_index(converter, "extant_sampling_probability")
    ].tolist() == pytest.approx([0.25] * engineer.num_time_bins)
    assert data.x[
        virtual_slice, _feature_index(converter, "is_virtual_node")
    ].tolist() == pytest.approx([1.0] * engineer.num_time_bins)
