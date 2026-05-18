"""Regression tests for scientific data-contract behavior."""

import pytest
import torch

from tests.support import require_modules

require_modules("torch", "torch_geometric", "ete3")

from ete3 import Tree  # noqa: E402
from torch_geometric.data import Data  # noqa: E402

from phylognn.data import TreeFeatureEngineer, TreeToGraphConverter  # noqa: E402


def _node_index(data, name):
    try:
        return data.node_names.index(name)
    except ValueError as exc:
        raise AssertionError(f"Node {name!r} not found") from exc


def _virtual_node_index(data, bin_label):
    return _node_index(data, f"__virtual_time_bin_{bin_label}__")


def _feature_index(converter, feature_name):
    try:
        return converter.output_feature_names.index(feature_name)
    except ValueError as exc:
        raise AssertionError(f"Feature {feature_name!r} not found") from exc


def _edge_pairs_by_type(data, edge_type):
    edge_mask = data.edge_type == edge_type
    typed_edges = data.edge_index[:, edge_mask].t().tolist()
    return {tuple(edge) for edge in typed_edges}


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
    """Repeated conversions should preserve virtual-node order and time labels."""
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
    assert torch.equal(first.time_bin, second.time_bin)
    assert first.time_bin[first.original_num_nodes :].tolist() == [0, 1, 2, 3]


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


def test_converter_rejects_integer_like_time_bins_for_virtual_nodes():
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

    with pytest.raises(ValueError, match="integer-like"):
        converter._add_virtual_nodes(data)


@pytest.mark.parametrize(
    ("bad_value", "match"),
    [
        (1.5, "integer-like"),
        (float("nan"), "non-finite"),
        (float("inf"), "non-finite"),
        (True, "boolean"),
    ],
)
def test_converter_rejects_invalid_requested_time_bin_labels(bad_value, match):
    """Requested time-bin labels must be finite non-boolean integer labels."""
    tree = Tree("(A:1,B:1)Root:0;", format=1)
    nodes = list(tree.traverse("preorder"))
    for node in nodes:
        node.time_bin = 0
    nodes[-1].time_bin = bad_value
    converter = TreeToGraphConverter(
        feature_names=("time_bin",),
        add_virtual_nodes=False,
        append_is_virtual_feature=False,
    )

    with pytest.raises(ValueError, match=match):
        converter.convert(tree)


def test_converter_rejects_generated_time_bin_graph_attrs():
    """The generated time-bin field name should not be caller metadata."""
    tree = Tree("(A:1,B:1)Root:0;", format=1)
    for node in tree.traverse("preorder"):
        node.time_bin = 0
        node.node_time = 0.0
    converter = TreeToGraphConverter(feature_names=("time_bin",))

    with pytest.raises(ValueError, match=r'graph_attrs\["time_bin"\].*generated graph fields'):
        converter.convert(tree, graph_attrs={"time_bin": "manual"})


def test_converter_allows_time_bin_graph_attrs_when_field_is_not_generated():
    """Caller metadata may use time_bin only when conversion does not generate it."""
    tree = Tree("(A:1,B:1)Root:0;", format=1)
    for node in tree.traverse("preorder"):
        node.node_time = 0.0
    converter = TreeToGraphConverter(feature_names=("node_time",))

    data = converter.convert(tree, graph_attrs={"time_bin": "manual"})

    assert data.time_bin == "manual"


def test_converter_missing_requested_time_bin_uses_required_feature_error():
    """Missing requested time-bin attributes should keep the required-feature path."""
    tree = Tree("(A:1,B:1)Root:0;", format=1)
    converter = TreeToGraphConverter(feature_names=("time_bin",))

    with pytest.raises(AttributeError, match="missing required attribute 'time_bin'"):
        converter.convert(tree)


def test_converter_non_numeric_requested_time_bin_uses_feature_type_error():
    """Non-numeric requested time-bin attributes should keep the type path."""
    tree = Tree("(A:1,B:1)Root:0;", format=1)
    for node in tree.traverse("preorder"):
        node.time_bin = 0
    next(tree.traverse("preorder")).time_bin = "bad"
    converter = TreeToGraphConverter(feature_names=("time_bin",))

    with pytest.raises(TypeError, match="Feature 'time_bin'.*must be numeric"):
        converter.convert(tree)


def test_rescaled_virtual_edges_follow_post_rescale_time_bins_and_metadata():
    """Virtual edges should connect original nodes by their post-rescale bin labels."""
    tree = Tree("((A:1,B:2)C:3,D:4)Root:0;", format=1)
    engineer = TreeFeatureEngineer(num_time_bins=5)
    tree = engineer.add_features(tree, origin_time=5.0, rescale=True)
    converter = TreeToGraphConverter(
        feature_names=engineer.feature_names,
        add_virtual_nodes=True,
        num_time_bins=engineer.num_time_bins,
        append_is_virtual_feature=True,
        preserve_node_names=True,
    )

    data = converter.convert(tree)

    assert data.num_time_bins == 5
    assert data.virtual_node_mask[: data.original_num_nodes].tolist() == [False] * 5
    assert data.virtual_node_mask[data.original_num_nodes :].tolist() == [True] * 5
    assert (
        data.node_type[: data.original_num_nodes].tolist()
        == [TreeToGraphConverter.NODE_TYPE_ORIGINAL] * 5
    )
    assert (
        data.node_type[data.original_num_nodes :].tolist()
        == [TreeToGraphConverter.NODE_TYPE_VIRTUAL] * 5
    )

    time_bin_idx = _feature_index(converter, "time_bin")
    expected_bins = {
        "Root": 4,
        "C": 2,
        "A": 1,
        "B": 0,
        "D": 1,
    }
    for node_name, bin_label in expected_bins.items():
        assert data.x[_node_index(data, node_name), time_bin_idx].item() == pytest.approx(
            float(bin_label)
        )

    virtual_edges = _edge_pairs_by_type(
        data,
        TreeToGraphConverter.EDGE_TYPE_VIRTUAL_TO_REAL,
    )
    expected_virtual_edges = set()
    for node_name, bin_label in expected_bins.items():
        original_idx = _node_index(data, node_name)
        virtual_idx = _virtual_node_index(data, bin_label)
        expected_virtual_edges.add((virtual_idx, original_idx))
        expected_virtual_edges.add((original_idx, virtual_idx))
    assert virtual_edges == expected_virtual_edges

    chain_edges = _edge_pairs_by_type(
        data,
        TreeToGraphConverter.EDGE_TYPE_VIRTUAL_CHAIN,
    )
    expected_chain_edges = set()
    for bin_label in range(engineer.num_time_bins - 1):
        left = _virtual_node_index(data, bin_label)
        right = _virtual_node_index(data, bin_label + 1)
        expected_chain_edges.add((left, right))
        expected_chain_edges.add((right, left))
    assert chain_edges == expected_chain_edges


def test_explicit_num_time_bins_is_authoritative_for_virtual_nodes():
    """Explicit num_time_bins should preserve empty bins above observed original bins."""
    converter = TreeToGraphConverter(
        feature_names=("time_bin",),
        add_virtual_nodes=True,
        num_time_bins=5,
        append_is_virtual_feature=False,
    )
    data = Data(
        x=torch.tensor([[0.0], [2.0]], dtype=torch.float32),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_type=torch.empty((0,), dtype=torch.long),
        original_num_nodes=2,
    )

    data = converter._add_virtual_nodes(data)

    assert data.num_time_bins == 5
    assert data.x.shape[0] == 7
    assert data.x[2:, 0].tolist() == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0])


def test_non_rescaled_virtual_edges_keep_original_time_bin_contract():
    """Non-rescaled virtual nodes should keep connecting by original timeline bins."""
    tree = Tree("((A:1,B:2)C:3,D:4)Root:0;", format=1)
    engineer = TreeFeatureEngineer(num_time_bins=5)
    tree = engineer.add_features(tree, origin_time=5.0, rescale=False)
    converter = TreeToGraphConverter(
        feature_names=engineer.feature_names,
        add_virtual_nodes=True,
        num_time_bins=engineer.num_time_bins,
        append_is_virtual_feature=True,
        preserve_node_names=True,
    )

    data = converter.convert(tree)

    assert data.num_time_bins == 5
    assert data.node_names[-5:] == [
        "__virtual_time_bin_0__",
        "__virtual_time_bin_1__",
        "__virtual_time_bin_2__",
        "__virtual_time_bin_3__",
        "__virtual_time_bin_4__",
    ]

    expected_bins = {
        "Root": 4,
        "C": 2,
        "A": 1,
        "B": 0,
        "D": 1,
    }
    virtual_edges = _edge_pairs_by_type(
        data,
        TreeToGraphConverter.EDGE_TYPE_VIRTUAL_TO_REAL,
    )
    expected_virtual_edges = set()
    for node_name, bin_label in expected_bins.items():
        original_idx = _node_index(data, node_name)
        virtual_idx = _virtual_node_index(data, bin_label)
        expected_virtual_edges.add((virtual_idx, original_idx))
        expected_virtual_edges.add((original_idx, virtual_idx))
    assert virtual_edges == expected_virtual_edges
