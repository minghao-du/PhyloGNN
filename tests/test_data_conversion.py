"""Focused tests for TreeToGraphConverter public contracts."""

import pytest

pytest.importorskip("ete3")
torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from ete3 import Tree  # noqa: E402
from torch_geometric.data import Data  # noqa: E402

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


@pytest.mark.parametrize("num_time_bins", [True, False, 5.0, "5"])
def test_converter_rejects_non_int_explicit_num_time_bins(num_time_bins):
    with pytest.raises(TypeError, match="num_time_bins"):
        TreeToGraphConverter(
            feature_names=("node_time", "time_bin"),
            add_virtual_nodes=True,
            num_time_bins=num_time_bins,
        )


@pytest.mark.parametrize("num_time_bins", [0, -1])
def test_converter_rejects_non_positive_explicit_num_time_bins(num_time_bins):
    with pytest.raises(ValueError, match="num_time_bins"):
        TreeToGraphConverter(
            feature_names=("node_time", "time_bin"),
            add_virtual_nodes=True,
            num_time_bins=num_time_bins,
        )


@pytest.mark.parametrize("num_time_bins", [1, 2, 5])
def test_converter_accepts_positive_int_explicit_num_time_bins(num_time_bins):
    converter = TreeToGraphConverter(
        feature_names=("node_time", "time_bin"),
        add_virtual_nodes=True,
        num_time_bins=num_time_bins,
    )

    assert converter.num_time_bins == num_time_bins


def test_converter_preserves_num_time_bins_inference_when_explicit_count_is_none():
    converter = TreeToGraphConverter(
        feature_names=("node_time", "time_bin"),
        add_virtual_nodes=True,
        num_time_bins=None,
    )

    assert converter.num_time_bins is None


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


def test_convert_rejects_graph_attrs_conflicting_with_base_generated_fields():
    """Caller metadata should not overwrite authoritative PyG fields."""
    tree = Tree("((A:1,B:2)C:3,D:4)Root:0;", format=1)
    engineer = TreeFeatureEngineer(num_time_bins=5)
    tree = engineer.add_features(tree, origin_time=5.0, rescale=False)
    converter = TreeToGraphConverter(feature_names=engineer.feature_names)

    with pytest.raises(ValueError) as exc_info:
        converter.convert(tree, graph_attrs={"edge_index": "bad", "x": "bad"})

    message = str(exc_info.value)
    assert 'graph_attrs["edge_index"]' in message
    assert 'graph_attrs["x"]' in message


def test_convert_preserves_safe_graph_attrs_without_changing_generated_fields():
    """Non-conflicting metadata should be attached unchanged."""
    tree = Tree("((A:1,B:2)C:3,D:4)Root:0;", format=1)
    engineer = TreeFeatureEngineer(num_time_bins=5)
    tree = engineer.add_features(tree, origin_time=5.0, rescale=False)
    converter = TreeToGraphConverter(
        feature_names=engineer.feature_names,
        add_virtual_nodes=True,
        num_time_bins=engineer.num_time_bins,
    )

    data = converter.convert(tree, graph_attrs={"dataset_id": "sim-001", "fold": 3})

    assert data.dataset_id == "sim-001"
    assert data.fold == 3
    assert torch.is_tensor(data.x)
    assert torch.is_tensor(data.edge_index)
    assert torch.is_tensor(data.edge_type)
    assert data.original_num_nodes == 5


@pytest.mark.parametrize(
    "generated_key",
    ["node_names", "num_time_bins", "time_bin", "virtual_node_mask", "node_type"],
)
def test_convert_rejects_graph_attrs_conflicting_with_optional_generated_fields(generated_key):
    """Dynamic conversion fields should be protected, not just base fields."""
    tree = Tree("((A:1,B:2)C:3,D:4)Root:0;", format=1)
    engineer = TreeFeatureEngineer(num_time_bins=5)
    tree = engineer.add_features(tree, origin_time=5.0, rescale=False)
    converter = TreeToGraphConverter(
        feature_names=engineer.feature_names,
        add_virtual_nodes=True,
        num_time_bins=engineer.num_time_bins,
        preserve_node_names=True,
    )

    with pytest.raises(ValueError, match=generated_key):
        converter.convert(tree, graph_attrs={generated_key: "bad"})


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


def test_load_data_uses_explicit_trusted_complete_object_load(tmp_path, monkeypatch):
    """Saved graph artifacts should opt into complete-object loading explicitly."""
    path = tmp_path / "graph.pt"
    graph = Data(
        x=torch.ones((1, 1)),
        edge_index=torch.empty((2, 0), dtype=torch.long),
    )
    calls = []

    def fake_load(load_path, *, map_location=None, weights_only=None):
        calls.append((load_path, map_location, weights_only))
        return graph

    monkeypatch.setattr(torch, "load", fake_load)

    loaded = TreeToGraphConverter.load_data(path, map_location="cpu")

    assert loaded is graph
    assert calls == [(path, "cpu", False)]


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


# ---------------------------------------------------------------------------
# Per-tree extant_sampling_probability converter pipeline tests (US2)
# ---------------------------------------------------------------------------


def test_converter_pipeline_with_per_tree_sampling_probability():
    """End-to-end: per-call extant_sampling_probability=0.7 flows into Data.x."""
    tree = Tree("((A:1,B:2)C:3,D:4)Root:0;", format=1)
    engineer = TreeFeatureEngineer(num_time_bins=5)
    tree = engineer.add_features(
        tree, origin_time=5.0, rescale=False, extant_sampling_probability=0.7
    )

    converter = TreeToGraphConverter(
        feature_names=engineer.feature_names,
        add_virtual_nodes=False,
    )
    data = converter.convert(tree)

    prob_idx = _feature_index(converter, "extant_sampling_probability")
    for i in range(data.num_nodes):
        assert data.x[i, prob_idx].item() == pytest.approx(0.7)


def test_converter_virtual_nodes_copy_per_tree_sampling_probability():
    """Virtual nodes receive the per-call value, not a stale constructor default."""
    tree = Tree("((A:1,B:2)C:3,D:4)Root:0;", format=1)
    engineer = TreeFeatureEngineer(
        num_time_bins=5,
        extant_sampling_probability=0.3,  # constructor default
    )
    tree = engineer.add_features(
        tree,
        origin_time=5.0,
        rescale=True,
        extant_sampling_probability=0.7,  # per-call override
    )

    converter = TreeToGraphConverter(
        feature_names=engineer.feature_names,
        add_virtual_nodes=True,
        num_time_bins=engineer.num_time_bins,
        append_is_virtual_feature=True,
        copy_sampling_prob_to_virtual=True,
    )
    data = converter.convert(tree)

    prob_idx = _feature_index(converter, "extant_sampling_probability")

    # All original nodes should have 0.7
    for i in range(data.original_num_nodes):
        assert data.x[i, prob_idx].item() == pytest.approx(0.7)

    # Virtual nodes should also have 0.7 (copied from first original node)
    virtual_slice = slice(data.original_num_nodes, None)
    for val in data.x[virtual_slice, prob_idx].tolist():
        assert val == pytest.approx(0.7)
