"""Focused tests for TreeFeatureEngineer public contracts."""

import pytest

pytest.importorskip("ete3")

from ete3 import Tree

from phylognn.data import TreeFeatureEngineer


def _node_by_name(tree, name):
    for node in tree.traverse():
        if node.name == name:
            return node
    raise AssertionError(f"Node {name!r} not found")


def _assert_temporal_features(
    node,
    *,
    node_time,
    time_bin,
    is_fossil=None,
    is_extant=None,
    is_sampled_ancestor=None,
    is_not_sampled_ancestor=None,
):
    assert node.node_time == pytest.approx(node_time)
    assert node.time_bin == time_bin

    if is_fossil is not None:
        assert node.is_fossil == is_fossil
    if is_extant is not None:
        assert node.is_extant == is_extant
    if is_sampled_ancestor is not None:
        assert node.is_sampled_ancestor == is_sampled_ancestor
    if is_not_sampled_ancestor is not None:
        assert node.is_not_sampled_ancestor == is_not_sampled_ancestor


def test_feature_metadata_views_are_read_only():
    """Public feature metadata should be immutable and deterministic."""
    engineer = TreeFeatureEngineer()

    assert isinstance(engineer.feature_names, tuple)
    assert isinstance(engineer.available_features, frozenset)
    assert engineer.feature_names == engineer.get_available_features()
    assert set(engineer.feature_names) == engineer.available_features


def test_custom_features_extend_public_metadata_consistently():
    """Custom features should appear in both public metadata views."""

    def add_dummy_feature(context):
        context["node"].add_feature("dummy_feature", 1.0)

    engineer = TreeFeatureEngineer(custom_features={"dummy_feature": add_dummy_feature})

    assert engineer.feature_names[-1] == "dummy_feature"
    assert "dummy_feature" in engineer.available_features


def test_add_features_accepts_read_only_feature_metadata():
    """Public feature metadata should work directly as converter input."""
    engineer = TreeFeatureEngineer(num_time_bins=5)
    tree = Tree("(A:1,B:1)Root;", format=1)

    tree = engineer.add_features(
        tree,
        origin_time=2.0,
        feature_names=engineer.feature_names[:3],
        rescale=False,
    )

    for node in tree.traverse():
        assert hasattr(node, "node_time")
        assert hasattr(node, "time_bin")
        assert hasattr(node, "is_internal")


def test_add_features_rejects_duplicate_feature_requests():
    """Feature selection should preserve a unique deterministic column order."""
    engineer = TreeFeatureEngineer()
    tree = Tree("(A:1,B:1)Root;", format=1)

    with pytest.raises(ValueError, match="must not contain duplicates"):
        engineer.add_features(
            tree,
            origin_time=2.0,
            feature_names=("node_time", "node_time"),
            rescale=False,
        )


def test_rescaled_features_use_effective_origin_time_for_temporal_features():
    """Rescaled branch lengths and temporal features should share one timeline."""
    tree = Tree("((A:1,B:2)C:3,D:4)Root:0;", format=1)
    engineer = TreeFeatureEngineer(num_time_bins=5)

    tree = engineer.add_features(tree, origin_time=5.0, rescale=True, inplace=False)

    assert _node_by_name(tree, "C").dist == pytest.approx(1.2)
    assert _node_by_name(tree, "A").dist == pytest.approx(0.4)
    assert _node_by_name(tree, "B").dist == pytest.approx(0.8)
    assert _node_by_name(tree, "D").dist == pytest.approx(1.6)

    for node in tree.traverse():
        assert node.rescale_factor == pytest.approx(0.4)

    _assert_temporal_features(
        _node_by_name(tree, "Root"),
        node_time=2.0,
        time_bin=4,
    )
    _assert_temporal_features(
        _node_by_name(tree, "C"),
        node_time=0.8,
        time_bin=2,
    )
    _assert_temporal_features(
        _node_by_name(tree, "A"),
        node_time=0.4,
        time_bin=1,
        is_fossil=1,
        is_extant=0,
        is_sampled_ancestor=0,
        is_not_sampled_ancestor=1,
    )
    _assert_temporal_features(
        _node_by_name(tree, "B"),
        node_time=0.0,
        time_bin=0,
        is_fossil=0,
        is_extant=1,
        is_sampled_ancestor=0,
        is_not_sampled_ancestor=0,
    )
    _assert_temporal_features(
        _node_by_name(tree, "D"),
        node_time=0.4,
        time_bin=1,
        is_fossil=1,
        is_extant=0,
    )


@pytest.mark.parametrize(
    (
        "newick",
        "origin_time",
        "num_time_bins",
        "expected_scale",
        "expected_effective_origin_time",
        "expected_interior_time",
        "expected_interior_bin",
    ),
    [
        ("(Present:4,Interior:1,Filler:1)Root:0;", 4.0, 5, 0.5, 2.0, 1.5, 3),
        ("(Present:2,Interior:0.5,Filler:0.5)Root:0;", 2.0, 5, 1.0, 2.0, 1.5, 3),
        ("(Present:0.3,Interior:0.1,Filler:0.2)Root:0;", 0.3, 7, 5.0, 1.5, 1.0, 4),
    ],
)
def test_rescaled_endpoint_bins_cover_scale_factor_classes(
    newick,
    origin_time,
    num_time_bins,
    expected_scale,
    expected_effective_origin_time,
    expected_interior_time,
    expected_interior_bin,
):
    """Endpoint and ceiling-rule bins should use the post-rescale origin time."""
    tree = Tree(newick, format=1)
    engineer = TreeFeatureEngineer(num_time_bins=num_time_bins)

    tree = engineer.add_features(tree, origin_time=origin_time, rescale=True)

    assert _node_by_name(tree, "Root").rescale_factor == pytest.approx(expected_scale)
    _assert_temporal_features(
        _node_by_name(tree, "Root"),
        node_time=expected_effective_origin_time,
        time_bin=num_time_bins - 1,
    )
    _assert_temporal_features(
        _node_by_name(tree, "Present"),
        node_time=0.0,
        time_bin=0,
        is_fossil=0,
        is_extant=1,
    )
    _assert_temporal_features(
        _node_by_name(tree, "Interior"),
        node_time=expected_interior_time,
        time_bin=expected_interior_bin,
        is_fossil=1,
        is_extant=0,
    )


def test_rescaled_zero_length_edges_remain_zero_on_effective_timeline():
    """Zero-length edges should stay zero while other temporal features rescale."""
    tree = Tree("(ZeroFossil:0,Present:2)Root:0;", format=1)
    engineer = TreeFeatureEngineer(num_time_bins=5)

    tree = engineer.add_features(tree, origin_time=2.0, rescale=True)

    zero_fossil = _node_by_name(tree, "ZeroFossil")
    assert zero_fossil.dist == 0.0
    _assert_temporal_features(
        zero_fossil,
        node_time=1.0,
        time_bin=4,
        is_fossil=1,
        is_extant=0,
        is_sampled_ancestor=1,
        is_not_sampled_ancestor=0,
    )
    _assert_temporal_features(
        _node_by_name(tree, "Present"),
        node_time=0.0,
        time_bin=0,
        is_fossil=0,
        is_extant=1,
    )


def test_rescale_rejects_tree_with_all_zero_branch_lengths():
    """Rescaling requires at least one non-zero branch length."""
    tree = Tree("(A:0,B:0)Root:0;", format=1)
    engineer = TreeFeatureEngineer(num_time_bins=5)

    with pytest.raises(ValueError, match="no non-zero branch lengths"):
        engineer.add_features(tree, origin_time=1.0, rescale=True)


def test_non_rescaled_temporal_features_keep_original_origin_time():
    """Non-rescaled workflows should keep the provided origin time unchanged."""
    tree = Tree("((A:1,B:2)C:3,D:4)Root:0;", format=1)
    engineer = TreeFeatureEngineer(num_time_bins=5)

    tree = engineer.add_features(tree, origin_time=5.0, rescale=False, inplace=False)

    assert _node_by_name(tree, "C").dist == pytest.approx(3.0)
    assert _node_by_name(tree, "A").dist == pytest.approx(1.0)
    assert _node_by_name(tree, "B").dist == pytest.approx(2.0)
    assert _node_by_name(tree, "D").dist == pytest.approx(4.0)

    for node in tree.traverse():
        assert node.rescale_factor == pytest.approx(1.0)

    _assert_temporal_features(
        _node_by_name(tree, "Root"),
        node_time=5.0,
        time_bin=4,
    )
    _assert_temporal_features(
        _node_by_name(tree, "C"),
        node_time=2.0,
        time_bin=2,
    )
    _assert_temporal_features(
        _node_by_name(tree, "A"),
        node_time=1.0,
        time_bin=1,
        is_fossil=1,
        is_extant=0,
    )
    _assert_temporal_features(
        _node_by_name(tree, "B"),
        node_time=0.0,
        time_bin=0,
        is_fossil=0,
        is_extant=1,
    )
    _assert_temporal_features(
        _node_by_name(tree, "D"),
        node_time=1.0,
        time_bin=1,
        is_fossil=1,
        is_extant=0,
    )
