"""Tests for single-tree region association evaluation."""

from dataclasses import FrozenInstanceError

import pytest
import torch


@pytest.mark.parametrize("tree", [None, "(A,B);", object()])
def test_build_leaf_laplacian_rejects_unsupported_tree_types(tree):
    """Only ETE trees can define a leaf constraint."""
    from phylognn.association import build_leaf_laplacian

    with pytest.raises(TypeError, match="tree"):
        build_leaf_laplacian(tree)


@pytest.mark.parametrize("names", [("", "B"), ("A", "A")])
def test_build_leaf_laplacian_rejects_blank_or_duplicate_tree_leaves(names):
    """Tree leaf names must be nonblank and unique before alignment."""
    from phylognn.association import build_leaf_laplacian

    tree = pytest.importorskip("ete3").Tree("(A:1,B:1);")
    first_leaf, second_leaf = tree.iter_leaves()
    first_leaf.name, second_leaf.name = names

    with pytest.raises(ValueError, match="Tree leaf names|Tree leaves"):
        build_leaf_laplacian(tree)


@pytest.mark.parametrize(
    "leaf_names",
    [
        ("A", "A", "C", "D", "E", "F"),
        ("A", "B", "C", "D", "E", "unknown"),
        ("A", "B", "C", "D", "E"),
    ],
)
def test_build_leaf_laplacian_rejects_invalid_explicit_leaf_names(association_tree, leaf_names):
    """Explicit orders must be complete unique permutations of tree leaves."""
    from phylognn.association import build_leaf_laplacian

    with pytest.raises(ValueError, match="leaf_names"):
        build_leaf_laplacian(association_tree, leaf_names)


@pytest.mark.parametrize("distance", [-1.0, float("nan"), float("inf")])
def test_build_leaf_laplacian_rejects_invalid_distances(association_tree, distance, monkeypatch):
    """Path distances must be finite and non-negative before construction."""
    from phylognn.association import build_leaf_laplacian

    leaf_type = type(next(association_tree.iter_leaves()))
    monkeypatch.setattr(leaf_type, "get_distance", lambda *_args, **_kwargs: distance)

    with pytest.raises(ValueError, match="distances"):
        build_leaf_laplacian(association_tree)


def test_build_leaf_laplacian_preserves_requested_leaf_order_and_properties(
    association_tree, torch_module
):
    """The normalized constraint is deterministic, symmetric, finite, and float32."""
    from phylognn.association import build_leaf_laplacian

    requested_names = ("F", "D", "A", "C", "E", "B")
    first = build_leaf_laplacian(association_tree, requested_names)
    second = build_leaf_laplacian(association_tree, requested_names)

    assert first.shape == (6, 6)
    assert first.dtype == torch_module.float32
    assert torch_module.isfinite(first).all()
    assert torch_module.allclose(first, first.T)
    assert torch_module.equal(first, second)


def test_build_leaf_laplacian_uses_topology_for_zero_path_lengths(torch_module):
    """An all-zero branch-length tree still yields nontrivial leaf constraints."""
    from phylognn.association import build_leaf_laplacian

    tree = pytest.importorskip("ete3").Tree("((A:0,B:0):0,(C:0,D:0):0);")
    laplacian = build_leaf_laplacian(tree)

    assert laplacian.shape == (4, 4)
    assert torch_module.isfinite(laplacian).all()
    assert not torch_module.allclose(laplacian, torch_module.eye(4))


def test_build_leaf_laplacian_accepts_a_single_leaf(torch_module):
    """A one-leaf normalized Laplacian is a valid finite scalar matrix."""
    from phylognn.association import build_leaf_laplacian

    tree = pytest.importorskip("ete3").Tree("A;")

    assert torch_module.equal(build_leaf_laplacian(tree), torch_module.ones((1, 1)))


def test_evaluate_region_association_returns_cv_scores_and_attention(
    association_leaf_names,
    association_position_mask,
    association_representations,
    association_targets,
    association_tree,
    torch_module,
):
    """CV/refit returns detached attention and scores without persistence."""
    from phylognn.association import evaluate_region_association

    result = evaluate_region_association(
        association_tree,
        association_representations,
        association_position_mask,
        association_targets,
        leaf_names=association_leaf_names,
        n_splits=3,
        epochs=8,
        hidden_dim=5,
        learning_rate=0.02,
        seed=7,
    )

    assert len(result.fold_r2) == 3
    assert all(torch_module.isfinite(torch_module.tensor(score)) for score in result.fold_r2)
    assert result.cv_r2 == pytest.approx(sum(result.fold_r2) / len(result.fold_r2))
    assert result.attention.shape == association_position_mask.shape
    assert torch_module.equal(
        result.attention[~association_position_mask],
        torch_module.zeros_like(result.attention[~association_position_mask]),
    )
    assert torch_module.allclose(result.attention.sum(dim=1), torch_module.ones(6))
    assert result.mean_attention.shape == (4,)
    assert result.attention.requires_grad is False
    assert result.mean_attention.requires_grad is False
    original_mean_attention = result.mean_attention.clone()
    result.attention[0, 0] = result.attention[0, 0] + 1.0
    assert torch_module.equal(result.mean_attention, original_mean_attention)

    with pytest.raises(FrozenInstanceError):
        result.cv_r2 = 0.0


def test_evaluate_region_association_accepts_leaf_name_targets(
    association_leaf_names,
    association_position_mask,
    association_representations,
    association_targets,
    association_tree,
):
    """Name-mapped targets respect explicit leaf alignment."""
    from phylognn.association import evaluate_region_association

    target_mapping = dict(zip(association_leaf_names, association_targets.tolist(), strict=True))
    result = evaluate_region_association(
        association_tree,
        association_representations,
        association_position_mask,
        target_mapping,
        n_splits=3,
        epochs=4,
        hidden_dim=4,
        seed=2,
    )

    assert len(result.fold_r2) == 3


@pytest.mark.parametrize(
    "case",
    [
        "non-tensor-representations",
        "wrong-representation-shape",
        "integer-representations",
        "nonfinite-representations",
        "wrong-mask-shape",
        "empty-mask-row",
        "non-tensor-targets",
        "wrong-target-shape",
        "nonfinite-targets",
    ],
)
def test_evaluate_region_association_rejects_malformed_tensors_before_training(
    association_position_mask,
    association_representations,
    association_targets,
    association_tree,
    case,
):
    """Input shapes, dtypes, finiteness, and masks are validated before fitting."""
    from phylognn.association import evaluate_region_association

    representations = association_representations.clone()
    position_mask = association_position_mask.clone()
    targets = association_targets.clone()
    expected_field = "representations"
    if case == "non-tensor-representations":
        representations = "not-a-tensor"
    elif case == "wrong-representation-shape":
        representations = representations[:, :, 0]
    elif case == "integer-representations":
        representations = representations.to(dtype=torch.long)
    elif case == "nonfinite-representations":
        representations[0, 0, 0] = float("nan")
    elif case == "wrong-mask-shape":
        position_mask = position_mask[:, :-1]
        expected_field = "position_mask"
    elif case == "empty-mask-row":
        position_mask[0] = False
        expected_field = "position_mask"
    elif case == "non-tensor-targets":
        targets = "not-targets"
        expected_field = "targets"
    elif case == "wrong-target-shape":
        targets = targets[:-1]
        expected_field = "targets"
    else:
        targets[0] = float("nan")
        expected_field = "targets"

    with pytest.raises((TypeError, ValueError), match=expected_field):
        evaluate_region_association(
            association_tree,
            representations,
            position_mask,
            targets,
            n_splits=3,
            epochs=1,
        )


def test_evaluate_region_association_rejects_target_mapping_key_mismatch(
    association_position_mask, association_representations, association_tree
):
    """Target mappings must neither omit nor introduce leaf names."""
    from phylognn.association import evaluate_region_association

    with pytest.raises(ValueError, match="Target mapping"):
        evaluate_region_association(
            association_tree,
            association_representations,
            association_position_mask,
            {"A": 0.0},
            n_splits=3,
            epochs=1,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_splits": True},
        {"n_splits": 1},
        {"n_splits": 4},
        {"epochs": True},
        {"epochs": 0},
        {"hidden_dim": 0},
        {"hidden_dim": 2.0},
        {"learning_rate": 0.0},
        {"learning_rate": float("inf")},
        {"weight_decay": -0.1},
        {"weight_decay": float("nan")},
        {"seed": 1.0},
    ],
)
def test_evaluate_region_association_rejects_invalid_hyperparameters(
    association_position_mask,
    association_representations,
    association_targets,
    association_tree,
    kwargs,
):
    """Fold and optimization settings are validated before model fitting."""
    from phylognn.association import evaluate_region_association

    with pytest.raises((TypeError, ValueError)):
        evaluate_region_association(
            association_tree,
            association_representations,
            association_position_mask,
            association_targets,
            **kwargs,
        )


def test_evaluate_region_association_rejects_constant_validation_targets(
    association_position_mask, association_representations, association_tree, torch_module
):
    """R-squared is rejected when a fold has zero target variance."""
    from phylognn.association import evaluate_region_association

    with pytest.raises(ValueError, match="constant targets"):
        evaluate_region_association(
            association_tree,
            association_representations,
            association_position_mask,
            torch_module.zeros(6),
            n_splits=3,
            epochs=1,
        )


def test_evaluate_region_association_rejects_nonfinite_fold_scores(
    association_position_mask,
    association_representations,
    association_targets,
    association_tree,
    monkeypatch,
    torch_module,
):
    """A non-finite validation R-squared must not be returned."""
    import phylognn.association as association

    def return_nonfinite_predictions(*args, **kwargs):
        return torch_module.full((6,), float("inf")), torch_module.zeros((6, 4))

    monkeypatch.setattr(association, "_fit_model", return_nonfinite_predictions)

    with pytest.raises(ValueError, match="R-squared"):
        association.evaluate_region_association(
            association_tree,
            association_representations,
            association_position_mask,
            association_targets,
            n_splits=3,
            epochs=1,
        )


def test_evaluate_region_association_is_repeatable_without_mutating_global_rng(
    association_position_mask,
    association_representations,
    association_targets,
    association_tree,
    torch_module,
):
    """A seeded call reproduces outputs while leaving the caller RNG untouched."""
    from phylognn.association import evaluate_region_association

    kwargs = {"n_splits": 3, "epochs": 3, "hidden_dim": 4, "seed": 11}
    torch_module.manual_seed(73)
    expected_next_values = torch_module.rand(4)
    torch_module.manual_seed(73)
    first = evaluate_region_association(
        association_tree,
        association_representations,
        association_position_mask,
        association_targets,
        **kwargs,
    )
    actual_next_values = torch_module.rand(4)
    second = evaluate_region_association(
        association_tree,
        association_representations,
        association_position_mask,
        association_targets,
        **kwargs,
    )

    assert torch_module.equal(actual_next_values, expected_next_values)
    assert first.fold_r2 == second.fold_r2
    assert torch_module.equal(first.attention, second.attention)
