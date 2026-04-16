"""Focused tests for TreeFeatureEngineer public contracts."""

import pytest


pytest.importorskip("ete3")

from ete3 import Tree

from phylognn.data import TreeFeatureEngineer


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
