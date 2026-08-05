"""Tests for single-tree region association evaluation."""

from dataclasses import FrozenInstanceError
import inspect
import random

import numpy as np
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


def test_prepare_region_association_preserves_default_tree_order_and_data_contract(
    association_leaf_names,
    association_position_mask,
    association_representations,
    association_targets,
    association_tree,
    torch_module,
):
    """Preparation stores the tree traversal order and all validated tensor fields."""
    from phylognn.association import prepare_region_association

    data = prepare_region_association(
        association_tree,
        association_representations,
        association_position_mask,
        association_targets,
    )

    assert data.leaf_names == association_leaf_names
    assert torch_module.equal(data.representations, association_representations)
    assert torch_module.equal(data.position_mask, association_position_mask)
    assert torch_module.equal(data.targets, association_targets)
    assert data.leaf_laplacian.shape == (6, 6)
    assert data.leaf_laplacian.dtype == torch_module.float32
    assert torch_module.isfinite(data.leaf_laplacian).all()
    with pytest.raises(FrozenInstanceError):
        data.leaf_names = association_leaf_names


def test_prepare_region_association_preserves_explicit_leaf_permutation(
    association_leaf_names,
    association_permuted_leaf_names,
    association_position_mask,
    association_representations,
    association_targets,
    association_tree,
    torch_module,
):
    """Explicit leaf order is accepted only as the caller's complete input alignment."""
    from phylognn.association import build_leaf_laplacian, prepare_region_association

    permutation = torch_module.tensor(
        [association_leaf_names.index(name) for name in association_permuted_leaf_names]
    )
    data = prepare_region_association(
        association_tree,
        association_representations[permutation],
        association_position_mask[permutation],
        association_targets[permutation],
        leaf_names=association_permuted_leaf_names,
    )

    assert data.leaf_names == association_permuted_leaf_names
    assert torch_module.equal(data.representations, association_representations[permutation])
    assert torch_module.equal(data.position_mask, association_position_mask[permutation])
    assert torch_module.equal(data.targets, association_targets[permutation])
    assert torch_module.equal(
        data.leaf_laplacian,
        build_leaf_laplacian(association_tree, association_permuted_leaf_names),
    )


def test_prepare_region_association_orders_mapped_targets_by_final_leaf_order(
    association_leaf_names,
    association_permuted_leaf_names,
    association_position_mask,
    association_representations,
    association_target_mapping,
    association_targets,
    association_tree,
    torch_module,
):
    """Name-keyed targets are reordered to the requested final leaf order only."""
    from phylognn.association import prepare_region_association

    permutation = torch_module.tensor(
        [association_leaf_names.index(name) for name in association_permuted_leaf_names]
    )
    data = prepare_region_association(
        association_tree,
        association_representations[permutation],
        association_position_mask[permutation],
        association_target_mapping,
        leaf_names=association_permuted_leaf_names,
    )

    assert torch_module.equal(data.targets, association_targets[permutation])


@pytest.mark.parametrize(
    "tree",
    [None, "(A,B);", object()],
)
def test_prepare_region_association_rejects_unsupported_tree_types(
    association_position_mask, association_representations, association_targets, tree
):
    """Preparation rejects inputs that cannot define a valid ETE leaf order."""
    from phylognn.association import prepare_region_association

    with pytest.raises(TypeError, match="tree"):
        prepare_region_association(
            tree, association_representations, association_position_mask, association_targets
        )


@pytest.mark.parametrize("names", [("", "B"), ("A", "A")])
def test_prepare_region_association_rejects_blank_or_duplicate_tree_leaf_names(
    association_position_mask, association_representations, association_targets, names
):
    """Preparation validates tree leaf names before accepting aligned tensors."""
    from phylognn.association import prepare_region_association

    tree = pytest.importorskip("ete3").Tree("(A:1,B:1);")
    first_leaf, second_leaf = tree.iter_leaves()
    first_leaf.name, second_leaf.name = names

    with pytest.raises(ValueError, match="Tree leaf names|Tree leaves"):
        prepare_region_association(
            tree,
            association_representations[:2],
            association_position_mask[:2],
            association_targets[:2],
        )


@pytest.mark.parametrize(
    "leaf_names",
    [
        ("A", "A", "C", "D", "E", "F"),
        ("A", "B", "C", "D", "E", "unknown"),
        ("A", "B", "C", "D", "E"),
    ],
)
def test_prepare_region_association_rejects_invalid_explicit_leaf_names(
    association_position_mask,
    association_representations,
    association_targets,
    association_tree,
    leaf_names,
):
    """Preparation rejects incomplete or non-unique explicit leaf orders."""
    from phylognn.association import prepare_region_association

    with pytest.raises(ValueError, match="leaf_names"):
        prepare_region_association(
            association_tree,
            association_representations,
            association_position_mask,
            association_targets,
            leaf_names=leaf_names,
        )


@pytest.mark.parametrize(
    "field",
    [
        "wrong-representation-shape",
        "integer-representations",
        "nonfinite-representations",
        "wrong-mask-shape",
        "empty-mask-row",
        "wrong-target-shape",
        "nonfinite-targets",
    ],
)
def test_prepare_region_association_rejects_malformed_tensor_contracts(
    association_position_mask,
    association_representations,
    association_targets,
    association_tree,
    field,
):
    """Preparation rejects shape, dtype, finiteness, and mask-contract violations."""
    from phylognn.association import prepare_region_association

    representations = association_representations.clone()
    position_mask = association_position_mask.clone()
    targets = association_targets.clone()
    expected_field = "representations"
    if field == "wrong-representation-shape":
        representations = representations[:, :, 0]
    elif field == "integer-representations":
        representations = representations.to(dtype=torch.long)
    elif field == "nonfinite-representations":
        representations[0, 0, 0] = float("nan")
    elif field == "wrong-mask-shape":
        position_mask = position_mask[:, :-1]
        expected_field = "position_mask"
    elif field == "empty-mask-row":
        position_mask[0] = False
        expected_field = "position_mask"
    elif field == "wrong-target-shape":
        targets = targets[:-1]
        expected_field = "targets"
    else:
        targets[0] = float("nan")
        expected_field = "targets"

    with pytest.raises((TypeError, ValueError), match=expected_field):
        prepare_region_association(association_tree, representations, position_mask, targets)


def test_prepare_region_association_rejects_target_mapping_key_mismatches(
    association_position_mask, association_representations, association_tree
):
    """Mapped targets must include every final leaf name exactly once."""
    from phylognn.association import prepare_region_association

    with pytest.raises(ValueError, match="Target mapping"):
        prepare_region_association(
            association_tree,
            association_representations,
            association_position_mask,
            {"A": 0.0},
        )


@pytest.mark.parametrize(
    "leaf_laplacian",
    [
        torch.ones((5, 5)),
        torch.ones((6, 6), dtype=torch.long),
        torch.full((6, 6), float("nan")),
    ],
)
def test_prepare_region_association_rejects_malformed_leaf_constraint_matrices(
    association_position_mask,
    association_representations,
    association_targets,
    association_tree,
    leaf_laplacian,
):
    """A supplied leaf constraint must be a finite float32-compatible [N, N] matrix."""
    from phylognn.association import prepare_region_association

    with pytest.raises((TypeError, ValueError), match="leaf_laplacian"):
        prepare_region_association(
            association_tree,
            association_representations,
            association_position_mask,
            association_targets,
            leaf_laplacian=leaf_laplacian,
        )


def test_prepare_region_association_accepts_non_laplacian_caller_constraint(
    association_position_mask,
    association_representations,
    association_targets,
    association_tree,
    torch_module,
):
    """Caller constraints need public tensor validity, not inferred Laplacian semantics."""
    from phylognn.association import prepare_region_association

    constraint = torch_module.arange(36, dtype=torch_module.float32).reshape(6, 6)
    data = prepare_region_association(
        association_tree,
        association_representations,
        association_position_mask,
        association_targets,
        leaf_laplacian=constraint,
    )

    assert torch_module.equal(data.leaf_laplacian, constraint)


def test_fit_region_association_returns_detached_all_leaf_results_for_selected_indices(
    association_position_mask,
    association_representations,
    association_targets,
    association_tree,
    torch_module,
):
    """A selected training subset still produces a complete detached fit result."""
    from phylognn.association import (
        RegionFitConfig,
        prepare_region_association,
        fit_region_association,
    )

    data = prepare_region_association(
        association_tree,
        association_representations,
        association_position_mask,
        association_targets,
    )
    train_indices = torch_module.tensor([1, 3, 5], dtype=torch_module.long)
    result = fit_region_association(
        data,
        train_indices=train_indices,
        config=RegionFitConfig(epochs=2, hidden_dim=4, seed=7),
    )

    assert torch_module.equal(result.train_indices, train_indices)
    assert result.predictions.shape == (6,)
    assert result.attention.shape == association_position_mask.shape
    assert len(result.losses) == 2
    assert all(np.isfinite(result.losses))
    assert not result.predictions.requires_grad
    assert not result.attention.requires_grad
    assert not result.train_indices.requires_grad
    assert torch_module.equal(
        result.attention[~association_position_mask],
        torch_module.zeros_like(result.attention[~association_position_mask]),
    )
    assert torch_module.equal(
        result.attention.sum(dim=1),
        torch_module.ones(6),
    )
    with pytest.raises(FrozenInstanceError):
        result.predictions = result.predictions


def test_fit_region_association_defaults_to_all_leaves_and_validated_config(
    association_position_mask,
    association_representations,
    association_targets,
    association_tree,
    torch_module,
):
    """Default settings retain the one-shot training defaults and all indices."""
    from phylognn.association import (
        RegionFitConfig,
        fit_region_association,
        prepare_region_association,
    )

    config = RegionFitConfig()
    assert config.epochs == 100
    assert config.learning_rate == pytest.approx(0.001)
    assert config.weight_decay == pytest.approx(0.0)
    assert config.hidden_dim == 32
    with pytest.raises(FrozenInstanceError):
        config.epochs = 1
    data = prepare_region_association(
        association_tree,
        association_representations,
        association_position_mask,
        association_targets,
    )
    result = fit_region_association(data, config=RegionFitConfig(epochs=1, seed=2))

    assert torch_module.equal(result.train_indices, torch_module.arange(6))


class _AssociationProbeModel(torch.nn.Module):
    """Small trainable model used to observe direct factory contracts."""

    def __init__(self, output_mode: str = "valid"):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))
        self.output_mode = output_mode

    def forward(self, representations, position_mask):
        predictions = self.bias.expand(representations.size(0))
        attention = position_mask.to(dtype=representations.dtype)
        if self.output_mode == "not-tuple":
            return predictions
        if self.output_mode == "prediction-shape":
            return predictions[:, None], attention
        if self.output_mode == "attention-shape":
            return predictions, attention[:, :-1]
        if self.output_mode == "nonfinite":
            return predictions + float("inf"), attention
        return predictions, attention


def _prepared_association_data(
    association_tree,
    association_representations,
    association_position_mask,
    association_targets,
):
    from phylognn.association import prepare_region_association

    return prepare_region_association(
        association_tree,
        association_representations,
        association_position_mask,
        association_targets,
    )


def test_fit_region_association_invokes_custom_factories_on_selected_targets(
    association_position_mask,
    association_representations,
    association_targets,
    association_tree,
    association_factory_probe,
    torch_module,
):
    """Custom factories are called directly and the loss sees only train targets."""
    from phylognn.association import RegionFitConfig, fit_region_association

    data = _prepared_association_data(
        association_tree,
        association_representations,
        association_position_mask,
        association_targets,
    )
    selected = torch_module.tensor([0, 2, 4])
    seen_targets = []

    def model_factory():
        association_factory_probe["model"].append(True)
        return _AssociationProbeModel()

    def loss_factory():
        association_factory_probe["loss"].append(True)

        def loss(predictions, targets):
            seen_targets.append(targets.detach().clone())
            return torch_module.mean((predictions - targets) ** 2)

        return loss

    def optimizer_factory(parameters):
        parameters = list(parameters)
        association_factory_probe["optimizer"].append(parameters)
        return torch_module.optim.SGD(parameters, lr=0.05)

    result = fit_region_association(
        data,
        train_indices=selected,
        config=RegionFitConfig(epochs=2, seed=3),
        model_factory=model_factory,
        loss_factory=loss_factory,
        optimizer_factory=optimizer_factory,
    )

    assert len(association_factory_probe["model"]) == 1
    assert len(association_factory_probe["loss"]) == 1
    assert len(association_factory_probe["optimizer"]) == 1
    assert len(seen_targets) == 2
    assert all(
        torch_module.equal(targets, association_targets[selected]) for targets in seen_targets
    )
    assert torch_module.equal(result.train_indices, selected)


@pytest.mark.parametrize(
    "output_mode",
    ["not-tuple", "prediction-shape", "attention-shape", "nonfinite"],
)
def test_fit_region_association_rejects_malformed_model_outputs_before_step(
    association_position_mask,
    association_representations,
    association_targets,
    association_tree,
    output_mode,
):
    """Malformed model output is rejected before training can step the optimizer."""
    from phylognn.association import RegionFitConfig, fit_region_association

    data = _prepared_association_data(
        association_tree,
        association_representations,
        association_position_mask,
        association_targets,
    )
    with pytest.raises((TypeError, ValueError), match="model|Model|attention|predictions"):
        fit_region_association(
            data,
            config=RegionFitConfig(epochs=1),
            model_factory=lambda: _AssociationProbeModel(output_mode),
        )


@pytest.mark.parametrize("loss_kind", ["not-tensor", "many-elements", "not-differentiable"])
def test_fit_region_association_rejects_invalid_loss_contracts_before_step(
    association_position_mask,
    association_representations,
    association_targets,
    association_tree,
    loss_kind,
):
    """Loss factories must return differentiable one-element tensors."""
    from phylognn.association import RegionFitConfig, fit_region_association

    data = _prepared_association_data(
        association_tree,
        association_representations,
        association_position_mask,
        association_targets,
    )

    def loss_factory():
        def loss(predictions, targets):
            if loss_kind == "not-tensor":
                return 1.0
            if loss_kind == "many-elements":
                return predictions - targets
            return torch.tensor(1.0)

        return loss

    with pytest.raises((TypeError, ValueError), match="loss|element|differentiable"):
        fit_region_association(
            data,
            config=RegionFitConfig(epochs=1),
            model_factory=_AssociationProbeModel,
            loss_factory=loss_factory,
        )


def test_fit_region_association_does_not_step_after_invalid_loss(
    association_position_mask,
    association_representations,
    association_targets,
    association_tree,
    torch_module,
):
    """Loss validation completes before the optimizer can mutate model state."""
    from phylognn.association import RegionFitConfig, fit_region_association

    data = _prepared_association_data(
        association_tree,
        association_representations,
        association_position_mask,
        association_targets,
    )
    step_calls = []

    class CountingSGD(torch_module.optim.SGD):
        def step(self, *args, **kwargs):
            step_calls.append(True)
            return super().step(*args, **kwargs)

    def optimizer_factory(parameters):
        return CountingSGD(list(parameters), lr=0.05)

    def loss_factory():
        return lambda _predictions, _targets: 1.0

    with pytest.raises(TypeError, match="loss"):
        fit_region_association(
            data,
            config=RegionFitConfig(epochs=1),
            model_factory=_AssociationProbeModel,
            loss_factory=loss_factory,
            optimizer_factory=optimizer_factory,
        )
    assert step_calls == []


def test_fit_region_association_rejects_optimizer_without_parameter_coverage(
    association_position_mask,
    association_representations,
    association_targets,
    association_tree,
    torch_module,
):
    """Every trainable model parameter must belong to the custom optimizer."""
    from phylognn.association import RegionFitConfig, fit_region_association

    class TwoParameterModel(_AssociationProbeModel):
        def __init__(self):
            super().__init__()
            self.extra = torch_module.nn.Parameter(torch_module.tensor(0.0))

    data = _prepared_association_data(
        association_tree,
        association_representations,
        association_position_mask,
        association_targets,
    )

    def optimizer_factory(parameters):
        parameters = list(parameters)
        return torch_module.optim.SGD([parameters[0]], lr=0.05)

    with pytest.raises(ValueError, match="every trainable model parameter"):
        fit_region_association(
            data,
            config=RegionFitConfig(epochs=1),
            model_factory=TwoParameterModel,
            optimizer_factory=optimizer_factory,
        )


def test_fit_region_association_selects_requested_device_and_restores_rng_state(
    association_position_mask,
    association_representations,
    association_targets,
    association_tree,
    torch_module,
):
    """Seeded CPU fitting is repeatable and leaves caller RNG streams unchanged."""
    from phylognn.association import RegionFitConfig, fit_region_association

    data = _prepared_association_data(
        association_tree,
        association_representations,
        association_position_mask,
        association_targets,
    )
    torch_module.manual_seed(41)
    expected_torch = torch_module.rand(3)
    torch_module.manual_seed(41)
    random.seed(41)
    expected_python = random.random()
    random.seed(41)
    np.random.seed(41)
    expected_numpy = np.random.rand()
    np.random.seed(41)

    seen_devices = []

    class DeviceProbeModel(_AssociationProbeModel):
        def forward(self, representations, position_mask):
            seen_devices.extend([representations.device, position_mask.device, self.bias.device])
            return super().forward(representations, position_mask)

    config = RegionFitConfig(epochs=2, seed=9, device="cpu")
    first = fit_region_association(data, config=config, model_factory=DeviceProbeModel)
    actual_torch = torch_module.rand(3)
    actual_python = random.random()
    actual_numpy = np.random.rand()
    second = fit_region_association(data, config=config, model_factory=DeviceProbeModel)

    assert torch_module.equal(actual_torch, expected_torch)
    assert actual_python == expected_python
    assert actual_numpy == expected_numpy
    assert all(device.type == "cpu" for device in seen_devices)
    assert torch_module.equal(first.predictions, second.predictions)
    assert torch_module.equal(first.attention, second.attention)
    assert first.losses == second.losses


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_fit_region_association_cuda_seed_does_not_change_caller_rng(
    association_position_mask,
    association_representations,
    association_targets,
    association_tree,
    torch_module,
):
    """Non-CPU seeded fitting isolates caller RNG when an accelerator exists."""
    from phylognn.association import RegionFitConfig, fit_region_association

    data = _prepared_association_data(
        association_tree,
        association_representations,
        association_position_mask,
        association_targets,
    )
    torch_module.manual_seed(17)
    expected = torch_module.rand(2)
    torch_module.manual_seed(17)
    actual = fit_region_association(data, config=RegionFitConfig(epochs=1, seed=8, device="cuda"))
    assert actual.predictions.device.type == "cuda"
    assert torch_module.equal(torch_module.rand(2), expected)


def test_cross_validate_region_association_is_seeded_and_assembles_complete_oof_results(
    association_position_mask,
    association_representations,
    association_targets,
    association_tree,
    torch_module,
):
    """Seeded CPU CV is repeatable, isolated, and covers every leaf once."""
    from phylognn.association import (
        RegionFitConfig,
        cross_validate_region_association,
    )

    data = _prepared_association_data(
        association_tree,
        association_representations,
        association_position_mask,
        association_targets,
    )
    config = RegionFitConfig(epochs=2, seed=13)
    torch_module.manual_seed(29)
    expected_torch = torch_module.rand(3)
    torch_module.manual_seed(29)
    random.seed(29)
    expected_python = random.random()
    random.seed(29)
    np.random.seed(29)
    expected_numpy = np.random.rand()
    np.random.seed(29)

    first = cross_validate_region_association(
        data,
        n_splits=3,
        config=config,
    )
    actual_torch = torch_module.rand(3)
    actual_python = random.random()
    actual_numpy = np.random.rand()
    second = cross_validate_region_association(
        data,
        n_splits=3,
        config=config,
    )

    assert torch_module.equal(actual_torch, expected_torch)
    assert actual_python == expected_python
    assert actual_numpy == expected_numpy
    assert all(
        torch_module.equal(first_fold, second_fold)
        for first_fold, second_fold in zip(
            first.validation_folds, second.validation_folds, strict=True
        )
    )
    assert first.fold_scores == second.fold_scores
    assert torch_module.equal(first.oof_predictions, second.oof_predictions)
    assert len(first.fold_results) == 3
    assert first.final_fit is not None
    assert first.oof_predictions.shape == (6,)
    assert not first.oof_predictions.requires_grad
    assert torch_module.equal(
        torch_module.sort(torch_module.cat(first.validation_folds)).values,
        torch_module.arange(6),
    )
    for fold, result in zip(first.validation_folds, first.fold_results, strict=True):
        assert torch_module.equal(first.oof_predictions[fold], result.predictions[fold])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cross_validate_region_association_cuda_seed_does_not_change_caller_rng(
    association_position_mask,
    association_representations,
    association_targets,
    association_tree,
    torch_module,
):
    """Non-CPU CV isolates caller RNG state when an accelerator exists."""
    from phylognn.association import (
        RegionFitConfig,
        cross_validate_region_association,
    )

    data = _prepared_association_data(
        association_tree,
        association_representations,
        association_position_mask,
        association_targets,
    )
    torch_module.manual_seed(31)
    expected = torch_module.rand(2)
    torch_module.manual_seed(31)
    random.seed(31)
    expected_python = random.random()
    random.seed(31)
    np.random.seed(31)
    expected_numpy = np.random.rand()
    np.random.seed(31)

    result = cross_validate_region_association(
        data,
        n_splits=3,
        config=RegionFitConfig(epochs=1, seed=5, device="cuda"),
        model_factory=_AssociationProbeModel,
    )

    assert result.oof_predictions.device.type == "cuda"
    assert torch_module.equal(torch_module.rand(2), expected)
    assert random.random() == expected_python
    assert np.random.rand() == expected_numpy


def test_cross_validate_region_association_preserves_manual_folds_and_supports_no_refit(
    association_position_mask,
    association_representations,
    association_targets,
    association_tree,
    torch_module,
):
    """Manual folds retain caller order and fill every OOF prediction once."""
    from phylognn.association import (
        RegionFitConfig,
        cross_validate_region_association,
    )

    data = _prepared_association_data(
        association_tree,
        association_representations,
        association_position_mask,
        association_targets,
    )
    manual_folds = ([5, 1], [3, 0], [4, 2])
    result = cross_validate_region_association(
        data,
        validation_folds=manual_folds,
        config=RegionFitConfig(epochs=1, seed=7),
        refit=False,
        model_factory=_AssociationProbeModel,
        score_fn=lambda predictions, targets: torch_module.mean(predictions - targets),
    )

    assert result.final_fit is None
    assert tuple(fold.tolist() for fold in result.validation_folds) == manual_folds
    assert len(result.fold_results) == len(manual_folds)
    assert len(result.fold_scores) == len(manual_folds)
    assert result.cv_score == pytest.approx(sum(result.fold_scores) / len(manual_folds))
    for fold, fit_result in zip(result.validation_folds, result.fold_results, strict=True):
        assert torch_module.equal(result.oof_predictions[fold], fit_result.predictions[fold])


@pytest.mark.parametrize("n_splits", [True, 1, 4, 5])
def test_cross_validate_region_association_rejects_invalid_default_split_counts(
    association_position_mask,
    association_representations,
    association_targets,
    association_tree,
    n_splits,
):
    """Generated folds require two through floor(N / 2) splits."""
    from phylognn.association import cross_validate_region_association

    data = _prepared_association_data(
        association_tree,
        association_representations,
        association_position_mask,
        association_targets,
    )
    with pytest.raises(ValueError, match="n_splits"):
        cross_validate_region_association(data, n_splits=n_splits)


@pytest.mark.parametrize(
    "validation_folds",
    [
        ([], [0, 1, 2, 3, 4, 5]),
        ([0, 1], [1, 2], [3, 4, 5]),
        ([0, 1], [2, 3], [4]),
        ([0, 1], [2, 3], [4, 6]),
    ],
)
def test_cross_validate_region_association_rejects_invalid_folds_before_fitting(
    association_position_mask,
    association_representations,
    association_targets,
    association_tree,
    monkeypatch,
    validation_folds,
):
    """Invalid manual folds never begin a fit or expose a partial result."""
    import phylognn.association as association

    data = _prepared_association_data(
        association_tree,
        association_representations,
        association_position_mask,
        association_targets,
    )
    fit_calls = []
    monkeypatch.setattr(
        association,
        "fit_region_association",
        lambda *_args, **_kwargs: fit_calls.append(True),
    )

    with pytest.raises((TypeError, ValueError), match="validation_folds|fold"):
        association.cross_validate_region_association(data, validation_folds=validation_folds)
    assert fit_calls == []


@pytest.mark.parametrize(
    "score_fn",
    [
        lambda _predictions, _targets: [0.5],
        lambda _predictions, _targets: torch.tensor([0.5, 0.2]),
        lambda _predictions, _targets: torch.tensor(0.5, requires_grad=True),
        lambda _predictions, _targets: float("nan"),
    ],
)
def test_cross_validate_region_association_rejects_invalid_custom_scores(
    association_position_mask,
    association_representations,
    association_targets,
    association_tree,
    score_fn,
):
    """Custom scorers must return finite detached scalar numeric values."""
    from phylognn.association import (
        RegionFitConfig,
        cross_validate_region_association,
    )

    data = _prepared_association_data(
        association_tree,
        association_representations,
        association_position_mask,
        association_targets,
    )
    with pytest.raises((TypeError, ValueError), match="score_fn|score"):
        cross_validate_region_association(
            data,
            n_splits=3,
            config=RegionFitConfig(epochs=1, seed=3),
            model_factory=_AssociationProbeModel,
            score_fn=score_fn,
        )


def test_cross_validate_region_association_rejects_constant_default_r2_targets(
    association_position_mask,
    association_representations,
    association_tree,
    torch_module,
):
    """Default R-squared rejects a validation fold with constant targets."""
    from phylognn.association import cross_validate_region_association, prepare_region_association

    data = prepare_region_association(
        association_tree,
        association_representations,
        association_position_mask,
        torch_module.zeros(6),
    )
    with pytest.raises(ValueError, match="constant targets"):
        cross_validate_region_association(data, n_splits=3)


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


def test_evaluate_region_association_preserves_legacy_signature_and_delegates_to_cv(
    association_leaf_names,
    association_position_mask,
    association_representations,
    association_targets,
    association_tree,
    monkeypatch,
    torch_module,
):
    """The facade keeps its contract and derives every field from one final fit."""
    import phylognn.association as association

    signature = inspect.signature(association.evaluate_region_association)
    assert list(signature.parameters) == [
        "tree",
        "representations",
        "position_mask",
        "targets",
        "leaf_names",
        "n_splits",
        "epochs",
        "hidden_dim",
        "learning_rate",
        "weight_decay",
        "seed",
    ]
    assert signature.parameters["n_splits"].default == 5
    assert signature.parameters["epochs"].default == 100
    assert signature.parameters["hidden_dim"].default == 32
    assert signature.parameters["learning_rate"].default == 1e-3
    assert signature.parameters["weight_decay"].default == 0.0
    assert signature.parameters["seed"].default == 0

    prepared = association.prepare_region_association(
        association_tree,
        association_representations,
        association_position_mask,
        association_targets,
        leaf_names=association_leaf_names,
    )
    final_attention = torch_module.full((6, 4), 0.25)
    final_fit = association.RegionFitResult(
        predictions=torch_module.zeros(6),
        attention=final_attention,
        train_indices=torch_module.arange(6),
        losses=(0.1,),
    )
    expected_cv = association.RegionAssociationCVResult(
        cv_score=0.75,
        fold_scores=(0.5, 1.0),
        oof_predictions=torch_module.zeros(6),
        validation_folds=(torch_module.tensor([0, 1, 2]), torch_module.tensor([3, 4, 5])),
        fold_results=(final_fit, final_fit),
        final_fit=final_fit,
    )
    cv_calls = []

    monkeypatch.setattr(
        association, "prepare_region_association", lambda *_args, **_kwargs: prepared
    )

    def fake_cross_validate(data, **kwargs):
        cv_calls.append((data, kwargs))
        return expected_cv

    monkeypatch.setattr(association, "cross_validate_region_association", fake_cross_validate)

    result = association.evaluate_region_association(
        association_tree,
        association_representations,
        association_position_mask,
        association_targets,
        leaf_names=association_leaf_names,
        n_splits=3,
        epochs=8,
        hidden_dim=5,
        learning_rate=0.02,
        weight_decay=0.1,
        seed=7,
    )

    assert len(cv_calls) == 1
    data, kwargs = cv_calls[0]
    assert data is prepared
    assert kwargs["n_splits"] == 3
    assert kwargs["refit"] is True
    assert kwargs["config"] == association.RegionFitConfig(
        epochs=8,
        hidden_dim=5,
        learning_rate=0.02,
        weight_decay=0.1,
        seed=7,
    )
    assert result.cv_r2 == expected_cv.cv_score
    assert result.fold_r2 == expected_cv.fold_scores
    assert torch_module.equal(result.attention, final_attention)
    assert torch_module.equal(result.mean_attention, final_attention.mean(dim=0))
    assert result.attention is not final_attention
    assert result.mean_attention.requires_grad is False


def test_evaluate_region_association_matches_equivalent_staged_workflow(
    association_position_mask,
    association_representations,
    association_targets,
    association_tree,
    torch_module,
):
    """The compatibility result is the CV result's one all-leaf final fit."""
    from phylognn.association import (
        RegionFitConfig,
        cross_validate_region_association,
        evaluate_region_association,
        prepare_region_association,
    )

    config = RegionFitConfig(epochs=3, hidden_dim=4, learning_rate=0.02, seed=17)
    data = prepare_region_association(
        association_tree,
        association_representations,
        association_position_mask,
        association_targets,
    )
    staged = cross_validate_region_association(data, n_splits=3, config=config, refit=True)
    result = evaluate_region_association(
        association_tree,
        association_representations,
        association_position_mask,
        association_targets,
        n_splits=3,
        epochs=config.epochs,
        hidden_dim=config.hidden_dim,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        seed=config.seed,
    )

    assert staged.final_fit is not None
    assert result.cv_r2 == staged.cv_score
    assert result.fold_r2 == staged.fold_scores
    assert torch_module.equal(result.attention, staged.final_fit.attention)
    assert torch_module.equal(result.mean_attention, staged.final_fit.attention.mean(dim=0))


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

    original_fit = association.fit_region_association

    def return_nonfinite_predictions(*args, **kwargs):
        result = original_fit(*args, **kwargs)
        return association.RegionFitResult(
            predictions=torch_module.full_like(result.predictions, float("inf")),
            attention=result.attention,
            train_indices=result.train_indices,
            losses=result.losses,
        )

    monkeypatch.setattr(association, "fit_region_association", return_nonfinite_predictions)

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
