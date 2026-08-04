"""Contract tests for node target attachment."""

import copy

import pytest

torch = pytest.importorskip("torch")
Data = pytest.importorskip("torch_geometric.data").Data
pytest.importorskip("ete3")

from phylognn.data import attach_node_targets  # noqa: E402


@pytest.fixture
def scalar_graph():
    """Return a deterministic graph with an internal node and two leaves."""
    return Data(
        x=torch.tensor([[1.0], [2.0], [3.0]]),
        edge_index=torch.tensor([[0, 0, 1, 2], [1, 2, 0, 0]], dtype=torch.long),
        node_names=["root", "A", "B"],
        metadata={"source": "fixture"},
    )


@pytest.fixture
def vector_graph(scalar_graph):
    """Return the scalar fixture for consistently shaped vector targets."""
    return scalar_graph


def test_aligns_field_records_in_node_name_order_and_ignores_extras(scalar_graph):
    result = attach_node_targets(
        scalar_graph,
        {
            "B": {"trait": 3.0},
            "extra": {"trait": 99.0},
            "root": {"trait": 1.0},
            "A": {"trait": 2.0},
        },
        target="trait",
    )

    assert torch.equal(result.y, torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(result.prediction_mask, torch.tensor([True, True, True]))
    assert scalar_graph.y is None


def test_aligns_direct_vector_values(vector_graph):
    result = attach_node_targets(
        vector_graph, {"A": [2.0, 3.0], "B": [4.0, 5.0], "root": [0.0, 1.0]}
    )

    assert result.y.shape == (3, 2)
    assert torch.equal(result.y[1], torch.tensor([2.0, 3.0]))


@pytest.mark.parametrize(
    "node_names", [None, ["A", "A", "B"], ["A", "", "B"], ["A", "   ", "B"], ["A", 2, "B"]]
)
def test_rejects_invalid_node_name_contract(scalar_graph, node_names):
    if node_names is None:
        del scalar_graph.node_names
    else:
        scalar_graph.node_names = node_names

    with pytest.raises((TypeError, ValueError), match="node_names"):
        attach_node_targets(scalar_graph, {"A": 1.0})


def test_mask_policy_handles_unselected_missing_and_nonfinite_values(scalar_graph):
    result = attach_node_targets(
        scalar_graph,
        {"A": 2.0, "B": float("inf")},
        node_selector=lambda _index, name: name != "root",
    )

    assert torch.isnan(result.y[0])
    assert torch.equal(result.prediction_mask, torch.tensor([False, True, False]))


def test_masked_nonfinite_vector_values_preserve_vector_target_shape(scalar_graph):
    result = attach_node_targets(
        scalar_graph,
        {
            "root": [float("nan"), 1.0],
            "A": [float("inf"), 2.0],
        },
    )

    assert result.y.shape == (3, 2)
    assert torch.isnan(result.y).all()
    assert not result.prediction_mask.any()


def test_masked_nonfinite_values_cannot_hide_shape_mismatches(scalar_graph):
    with pytest.raises(ValueError, match="shape"):
        attach_node_targets(
            scalar_graph,
            {
                "root": 1.0,
                "A": [float("nan"), 2.0],
            },
        )


def test_error_policy_reports_first_selected_missing_node(scalar_graph):
    with pytest.raises(ValueError, match="root"):
        attach_node_targets(scalar_graph, {"B": 3.0}, missing="error")


def test_empty_records_fall_back_to_scalar_nan_targets(scalar_graph):
    result = attach_node_targets(scalar_graph, {})

    assert result.y.shape == (3,)
    assert torch.isnan(result.y).all()
    assert not result.prediction_mask.any()


@pytest.mark.parametrize(
    ("records", "target"),
    [
        ({"A": {"trait": 1.0}}, None),
        ({"A": 1.0}, "trait"),
        ({"A": {"trait": 1.0}, "B": 2.0}, "trait"),
    ],
)
def test_rejects_mapping_mode_mismatches(scalar_graph, records, target):
    with pytest.raises((TypeError, ValueError)):
        attach_node_targets(scalar_graph, records, target=target)


def test_rejects_invalid_values_shapes_and_selector_results(scalar_graph):
    with pytest.raises((TypeError, ValueError), match="A"):
        attach_node_targets(scalar_graph, {"A": "not-a-number"})
    with pytest.raises(ValueError, match="shape"):
        attach_node_targets(scalar_graph, {"root": 1.0, "A": [2.0, 3.0]})
    with pytest.raises(TypeError, match="selector"):
        attach_node_targets(scalar_graph, {"A": 1.0}, node_selector=lambda *_args: 1)
    with pytest.raises(ValueError, match="missing"):
        attach_node_targets(scalar_graph, {"A": 1.0}, missing="ignore")
    with pytest.raises(ValueError, match="different"):
        attach_node_targets(
            scalar_graph, {"A": 1.0}, target_field="same", prediction_mask_field="same"
        )


def test_custom_output_fields_and_selector_exception_context(scalar_graph):
    result = attach_node_targets(
        scalar_graph,
        {"A": 2.0},
        target_field="trait_target",
        prediction_mask_field="trait_mask",
    )
    assert torch.isnan(result.trait_target[[0, 2]]).all()
    assert result.trait_target[1].item() == 2.0
    assert result.trait_mask.tolist() == [False, True, False]

    def failing_selector(index, _name):
        if index == 1:
            raise RuntimeError("boom")
        return True

    with pytest.raises(ValueError, match="node 1.*A"):
        attach_node_targets(
            scalar_graph,
            {"A": 2.0},
            node_selector=failing_selector,
        )


@pytest.mark.parametrize("field_name", ["trait-target", "trait target", "class", "__dict__"])
def test_rejects_non_addressable_output_fields_without_mutation(scalar_graph, field_name):
    with pytest.raises((TypeError, ValueError), match="field|attribute"):
        attach_node_targets(scalar_graph, {"A": 2.0}, target_field=field_name)
    assert not hasattr(scalar_graph, "y") or scalar_graph.y is None


@pytest.mark.parametrize("parameter", ["target_field", "prediction_mask_field"])
@pytest.mark.parametrize(
    "field_name",
    ["x", "edge_index", "node_names", "num_nodes", "batch", "original_num_nodes"],
)
def test_rejects_graph_structural_output_fields(scalar_graph, parameter, field_name):
    kwargs = {parameter: field_name}
    with pytest.raises(ValueError, match="structural"):
        attach_node_targets(scalar_graph, {"A": 2.0}, **kwargs)
    assert scalar_graph.x.shape == (3, 1)
    assert scalar_graph.edge_index.shape == (2, 4)
    assert scalar_graph.node_names == ["root", "A", "B"]


@pytest.mark.parametrize("parameter", ["target_field", "prediction_mask_field"])
@pytest.mark.parametrize("field_name", ["keys", "clone"])
def test_rejects_data_public_api_output_field_collisions(scalar_graph, parameter, field_name):
    with pytest.raises(ValueError, match="public Data API"):
        attach_node_targets(scalar_graph, {"A": 2.0}, **{parameter: field_name})

    assert callable(scalar_graph.keys)
    assert callable(scalar_graph.clone)
    assert scalar_graph.x.shape == (3, 1)
    assert scalar_graph.edge_index.shape == (2, 4)


def test_valid_configured_output_fields_remain_addressable(scalar_graph):
    result = attach_node_targets(
        scalar_graph,
        {"A": 2.0},
        target_field="trait_target",
        prediction_mask_field="trait_mask",
    )

    assert result.trait_target[1].item() == 2.0
    assert result.trait_mask.tolist() == [False, True, False]
    assert callable(result.keys)
    assert callable(result.clone)
    assert result.x.shape == (3, 1)
    assert result.edge_index.shape == (2, 4)


def test_inplace_mode_is_atomic_and_default_is_deeply_independent(scalar_graph):
    original = copy.deepcopy(scalar_graph)
    result = attach_node_targets(scalar_graph, {"root": 1.0, "A": 2.0, "B": 3.0})

    assert result is not scalar_graph
    result.metadata["source"] = "changed"
    assert scalar_graph.metadata == original.metadata

    with pytest.raises((TypeError, ValueError)):
        attach_node_targets(scalar_graph, {"A": "bad"}, inplace=True)
    assert scalar_graph.y is None

    returned = attach_node_targets(scalar_graph, {"root": 1.0}, inplace=True)
    assert returned is scalar_graph
    assert scalar_graph.prediction_mask.tolist() == [True, False, False]


def test_large_shuffled_alignment_preserves_graph_fields():
    node_count = 1_000
    names = [f"node-{index}" for index in range(node_count)]
    graph = Data(
        x=torch.arange(node_count, dtype=torch.float32).reshape(-1, 1),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        node_names=names,
        original_num_nodes=node_count,
    )
    records = {name: float(index) for index, name in reversed(list(enumerate(names)))}

    result = attach_node_targets(graph, records)

    assert torch.equal(result.y, torch.arange(node_count, dtype=torch.float32))
    assert result.node_names == names
    assert torch.equal(result.x, graph.x)
    assert torch.equal(result.edge_index, graph.edge_index)
    assert result.original_num_nodes == node_count


def test_converter_graph_targets_are_consumable_by_node_regression_validation():
    """Converted graphs retain aligned target fields required by node regression."""
    from ete3 import Tree

    from examples.extant_trait_regression import validate_graph_data
    from phylognn.data import TreeFeatureEngineer, TreeToGraphConverter

    tree = Tree("(A:1.0,B:1.0)root:0.0;", format=1)
    engineer = TreeFeatureEngineer()
    engineer.add_features(
        tree,
        origin_time=1.0,
        feature_names=["branch_length"],
        rescale=False,
        inplace=True,
    )
    graph = TreeToGraphConverter(
        feature_names=["branch_length"],
        append_is_virtual_feature=False,
    ).convert(tree)

    result = attach_node_targets(graph, {"B": 3.0, "root": 1.0, "A": 2.0})
    result.train_mask = torch.tensor([True, False, False])
    result.val_mask = torch.tensor([False, True, False])
    result.test_mask = torch.tensor([False, False, True])

    assert result.node_names == ["root", "A", "B"]
    assert torch.equal(result.y, torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(result.prediction_mask, torch.tensor([True, True, True]))
    validate_graph_data(result)
