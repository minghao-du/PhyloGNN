"""Focused tests for TreeToGraphConverter public contracts."""

import pytest

pytest.importorskip("ete3")
torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from ete3 import Tree  # noqa: E402

from phylognn.data import TreeFeatureEngineer, TreeToGraphConverter  # noqa: E402


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


def test_convert_generates_node_aligned_time_bin_field():
    """Requested time-bin features should also be exposed as node labels."""
    tree = Tree("((A:1,B:2)C:3,D:4)Root:0;", format=1)
    engineer = TreeFeatureEngineer(num_time_bins=5)
    tree = engineer.add_features(tree, origin_time=5.0, rescale=False)
    converter = TreeToGraphConverter(
        feature_names=engineer.feature_names,
        add_virtual_nodes=False,
    )

    data = converter.convert(tree)

    time_bin_idx = _feature_index(converter, "time_bin")
    assert torch.is_tensor(data.time_bin)
    assert data.time_bin.dtype == torch.long
    assert data.time_bin.shape == (data.num_nodes,)
    assert data.time_bin.tolist() == data.x[:, time_bin_idx].long().tolist()


def test_convert_and_save_preserves_generated_time_bin(tmp_path):
    """Generated node labels should survive the existing PyG Data persistence path."""
    tree = Tree("((A:1,B:2)C:3,D:4)Root:0;", format=1)
    engineer = TreeFeatureEngineer(num_time_bins=5)
    tree = engineer.add_features(tree, origin_time=5.0, rescale=False)
    converter = TreeToGraphConverter(
        feature_names=engineer.feature_names,
        add_virtual_nodes=False,
    )
    path = tmp_path / "graph.pt"

    saved = converter.convert_and_save(tree, path)
    loaded = TreeToGraphConverter.load_data(path)

    assert torch.equal(loaded.time_bin, saved.time_bin)


def test_convert_does_not_generate_time_bin_when_feature_is_absent():
    """Converters should not infer temporal labels from unrelated features."""
    tree = Tree("((A:1,B:2)C:3,D:4)Root:0;", format=1)
    engineer = TreeFeatureEngineer(num_time_bins=5)
    tree = engineer.add_features(tree, origin_time=5.0, rescale=False)
    converter = TreeToGraphConverter(
        feature_names=("node_time", "is_tip"),
        add_virtual_nodes=False,
    )

    data = converter.convert(tree)

    assert not hasattr(data, "time_bin")


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
    time_bin_idx = _feature_index(converter, "time_bin")
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
    assert data.time_bin.dtype == torch.long
    assert data.time_bin.shape == (data.num_nodes,)
    assert (
        data.time_bin[: data.original_num_nodes].tolist()
        == data.x[: data.original_num_nodes, time_bin_idx].long().tolist()
    )
    assert data.time_bin[virtual_slice].tolist() == [0, 1, 2, 3, 4]
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
