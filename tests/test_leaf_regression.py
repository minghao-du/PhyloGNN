"""Tests for the recommended leaf-regression workflow."""

import random

import numpy as np
import pytest
import torch


class _ConfiguredRegressor(torch.nn.Module):
    instances: list["_ConfiguredRegressor"] = []

    def __init__(self, offset: float) -> None:
        super().__init__()
        self.offset = offset
        self.weight = torch.nn.Parameter(torch.tensor(0.1))
        type(self).instances.append(self)

    def forward(
        self, representations: torch.Tensor, position_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        predictions = representations[:, 0, 0] * self.weight + self.offset
        attention = position_mask.to(dtype=representations.dtype)
        attention = attention / attention.sum(dim=1, keepdim=True)
        return predictions, attention


class _PredictionOnlyRegressor(torch.nn.Module):
    """Small custom model that intentionally omits interpretation output."""

    def __init__(self, offset: float) -> None:
        super().__init__()
        self.offset = offset
        self.weight = torch.nn.Parameter(torch.tensor(0.1))

    def forward(self, representations: torch.Tensor, position_mask: torch.Tensor) -> torch.Tensor:
        del position_mask
        return representations[:, 0, 0] * self.weight + self.offset


class _NoParameterRegressor(torch.nn.Module):
    """Return valid predictions without offering any trainable parameter."""

    def forward(self, representations: torch.Tensor, position_mask: torch.Tensor) -> torch.Tensor:
        del position_mask
        return representations[:, 0, 0]


class _InvalidOutputRegressor(torch.nn.Module):
    """Produce configurable invalid outputs while retaining a trainable probe."""

    instances: list["_InvalidOutputRegressor"] = []

    def __init__(self, kind: str) -> None:
        super().__init__()
        self.kind = kind
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        type(self).instances.append(self)

    def forward(self, representations: torch.Tensor, position_mask: torch.Tensor) -> object:
        predictions = representations[:, 0, 0] * self.weight
        if self.kind == "shape":
            return predictions[:, None]
        if self.kind == "nonfinite":
            return predictions * float("nan")
        if self.kind == "nondifferentiable":
            return predictions.detach()
        if self.kind == "attention":
            return predictions, position_mask[:, :-1].to(dtype=predictions.dtype)
        return (predictions,)


class _ConstructorFailureRegressor(torch.nn.Module):
    """Represent a model class that cannot satisfy its construction contract."""

    def __init__(self) -> None:
        raise RuntimeError("construction failed")


def _fold_size_score(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Return the held-out fold size for deterministic CV assertions."""
    del targets
    return float(predictions.numel())


def test_run_leaf_regression_uses_default_model_and_returns_complete_results(
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
):
    """The default model completes CV and one final refit without file output."""
    from phylognn import LeafRegressionConfig, run_leaf_regression

    result = run_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
        n_splits=3,
        training_config=LeafRegressionConfig(epochs=2, learning_rate=0.01, seed=17),
    )

    assert len(result.fold_scores) == 3
    assert torch.isfinite(result.oof_predictions).all()
    assert torch.isfinite(result.predictions).all()
    assert result.oof_predictions.shape == leaf_regression_targets.shape
    assert result.attention is not None
    assert result.mean_attention is not None


def test_run_leaf_regression_constructs_fresh_configured_models_and_weights_scores(
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
):
    """Every fold and refit receives a new class-plus-keyword model instance."""
    from phylognn import LeafRegressionConfig, run_leaf_regression

    _ConfiguredRegressor.instances.clear()
    folds = ([0, 1, 2], [3, 4], [5])
    result = run_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
        validation_folds=folds,
        training_config=LeafRegressionConfig(epochs=1, seed=7),
        score_fn=lambda predictions, targets: float(targets.numel()),
        model_class=_ConfiguredRegressor,
        model_config={"offset": 0.5},
    )

    assert len(_ConfiguredRegressor.instances) == len(folds) + 1
    assert len({id(model) for model in _ConfiguredRegressor.instances}) == len(folds) + 1
    assert all(model.offset == 0.5 for model in _ConfiguredRegressor.instances)
    assert result.fold_scores == (3.0, 2.0, 1.0)
    assert result.cv_score == pytest.approx((3.0 * 3 + 2.0 * 2 + 1.0) / 6)
    assert torch.isfinite(result.oof_predictions).all()


def test_run_leaf_regression_is_repeatable_and_restores_caller_rng_states(
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
):
    """A successful seeded workflow leaves Python, NumPy, and PyTorch RNG untouched."""
    from phylognn import LeafRegressionConfig, run_leaf_regression

    kwargs = {
        "n_splits": 3,
        "training_config": LeafRegressionConfig(epochs=1, seed=23),
    }
    random.seed(101)
    np.random.seed(101)
    torch.manual_seed(101)
    expected_python = random.random()
    expected_numpy = np.random.rand()
    expected_torch = torch.rand(3)
    random.seed(101)
    np.random.seed(101)
    torch.manual_seed(101)

    first = run_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
        **kwargs,
    )
    actual_python = random.random()
    actual_numpy = np.random.rand()
    actual_torch = torch.rand(3)
    second = run_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
        **kwargs,
    )

    assert actual_python == expected_python
    assert actual_numpy == expected_numpy
    assert torch.equal(actual_torch, expected_torch)
    assert first.fold_scores == second.fold_scores
    assert torch.equal(first.oof_predictions, second.oof_predictions)
    assert torch.equal(first.predictions, second.predictions)


def test_prepare_leaf_regression_preserves_rows_and_aligns_mapped_targets(
    leaf_regression_permuted_leaf_names,
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_target_mapping,
    leaf_regression_targets,
    leaf_regression_tree,
):
    """Explicit leaf order aligns mapped targets without reordering caller rows."""
    from phylognn import prepare_leaf_regression

    representations = leaf_regression_representations.clone()
    position_mask = leaf_regression_position_mask.clone()
    data = prepare_leaf_regression(
        leaf_regression_tree,
        representations,
        position_mask,
        leaf_regression_target_mapping,
        leaf_names=leaf_regression_permuted_leaf_names,
    )

    expected_targets = torch.tensor(
        [leaf_regression_target_mapping[name] for name in leaf_regression_permuted_leaf_names],
        dtype=torch.float32,
    )
    assert data.leaf_names == leaf_regression_permuted_leaf_names
    assert torch.equal(data.targets, expected_targets)
    assert torch.equal(data.representations, representations)
    assert torch.equal(data.position_mask, position_mask)
    assert torch.equal(representations, leaf_regression_representations)
    assert torch.equal(position_mask, leaf_regression_position_mask)
    assert not torch.equal(data.targets, leaf_regression_targets)


def test_fit_leaf_regression_returns_selected_indices_all_predictions_and_losses(
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
):
    """A staged fit trains selected leaves yet evaluates every prepared leaf."""
    from phylognn import LeafRegressionConfig, fit_leaf_regression, prepare_leaf_regression

    data = prepare_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
    )
    selected_indices = [5, 1, 3]
    result = fit_leaf_regression(
        data,
        train_indices=selected_indices,
        training_config=LeafRegressionConfig(epochs=3, learning_rate=0.01, seed=19),
        model_class=_ConfiguredRegressor,
        model_config={"offset": 0.25},
    )

    assert torch.equal(result.train_indices, torch.tensor(selected_indices))
    assert result.predictions.shape == (len(data.leaf_names),)
    assert result.attention is not None
    assert result.attention.shape == data.position_mask.shape
    assert len(result.losses) == 3
    assert all(np.isfinite(loss) for loss in result.losses)


def test_prediction_only_model_propagates_none_attention_across_workflows(
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
):
    """Prediction-only models never receive synthesized attention values."""
    from phylognn import (
        LeafRegressionConfig,
        cross_validate_leaf_regression,
        fit_leaf_regression,
        prepare_leaf_regression,
        run_leaf_regression,
    )

    config = LeafRegressionConfig(epochs=1, learning_rate=0.01, seed=37)
    model_kwargs = {"model_class": _PredictionOnlyRegressor, "model_config": {"offset": 0.2}}
    data = prepare_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
    )

    fit_result = fit_leaf_regression(data, training_config=config, **model_kwargs)
    cv_result = cross_validate_leaf_regression(
        data,
        validation_folds=([0, 1], [2, 3], [4, 5]),
        training_config=config,
        score_fn=_fold_size_score,
        **model_kwargs,
    )
    workflow_result = run_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
        validation_folds=([0, 1], [2, 3], [4, 5]),
        training_config=config,
        score_fn=_fold_size_score,
        **model_kwargs,
    )

    assert fit_result.attention is None
    assert all(result.attention is None for result in cv_result.fold_results)
    assert cv_result.final_fit is not None
    assert cv_result.final_fit.attention is None
    assert workflow_result.attention is None
    assert workflow_result.mean_attention is None


def test_attention_model_preserves_masked_attention_across_workflows(
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
):
    """Attention-producing models retain real masked attention at every stage."""
    from phylognn import (
        LeafRegressionConfig,
        cross_validate_leaf_regression,
        fit_leaf_regression,
        prepare_leaf_regression,
        run_leaf_regression,
    )

    config = LeafRegressionConfig(epochs=1, learning_rate=0.01, seed=41)
    model_kwargs = {"model_class": _ConfiguredRegressor, "model_config": {"offset": 0.2}}
    data = prepare_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
    )

    fit_result = fit_leaf_regression(data, training_config=config, **model_kwargs)
    cv_result = cross_validate_leaf_regression(
        data,
        validation_folds=([0, 1], [2, 3], [4, 5]),
        training_config=config,
        score_fn=_fold_size_score,
        **model_kwargs,
    )
    workflow_result = run_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
        validation_folds=([0, 1], [2, 3], [4, 5]),
        training_config=config,
        score_fn=_fold_size_score,
        **model_kwargs,
    )

    assert fit_result.attention is not None
    assert fit_result.attention.shape == data.position_mask.shape
    assert torch.all(fit_result.attention[~data.position_mask] == 0)
    assert all(result.attention is not None for result in cv_result.fold_results)
    assert cv_result.final_fit is not None
    assert cv_result.final_fit.attention is not None
    assert workflow_result.attention is not None
    assert workflow_result.mean_attention is not None
    assert workflow_result.mean_attention.shape == (data.representations.size(1),)
    assert torch.all(workflow_result.attention[~data.position_mask] == 0)


def test_cross_validation_preserves_manual_folds_and_matches_recommended_workflow(
    leaf_regression_permuted_leaf_names,
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_target_mapping,
    leaf_regression_tree,
):
    """Manual folds retain caller order, cover OOF output, and match the facade."""
    from phylognn import (
        LeafRegressionConfig,
        cross_validate_leaf_regression,
        prepare_leaf_regression,
        run_leaf_regression,
    )

    folds = ([5], [2, 0], [4, 3, 1])
    config = LeafRegressionConfig(epochs=2, learning_rate=0.01, seed=29)
    model_config = {"offset": 0.75}
    data = prepare_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_target_mapping,
        leaf_names=leaf_regression_permuted_leaf_names,
    )

    without_refit = cross_validate_leaf_regression(
        data,
        n_splits=99,
        validation_folds=folds,
        training_config=config,
        score_fn=_fold_size_score,
        refit=False,
        model_class=_ConfiguredRegressor,
        model_config=model_config,
    )
    with_refit = cross_validate_leaf_regression(
        data,
        n_splits=99,
        validation_folds=folds,
        training_config=config,
        score_fn=_fold_size_score,
        model_class=_ConfiguredRegressor,
        model_config=model_config,
    )
    recommended = run_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_target_mapping,
        leaf_names=leaf_regression_permuted_leaf_names,
        n_splits=99,
        validation_folds=folds,
        training_config=config,
        score_fn=_fold_size_score,
        model_class=_ConfiguredRegressor,
        model_config=model_config,
    )

    assert tuple(fold.tolist() for fold in without_refit.validation_folds) == tuple(folds)
    assert torch.isfinite(without_refit.oof_predictions).all()
    assert torch.equal(
        torch.sort(torch.cat(without_refit.validation_folds)).values,
        torch.arange(len(data.leaf_names), dtype=torch.long),
    )
    assert without_refit.final_fit is None
    assert with_refit.final_fit is not None
    assert torch.equal(
        with_refit.final_fit.train_indices,
        torch.arange(len(data.leaf_names), dtype=torch.long),
    )
    assert torch.equal(with_refit.oof_predictions, recommended.oof_predictions)
    assert with_refit.fold_scores == recommended.fold_scores
    assert with_refit.cv_score == recommended.cv_score
    assert torch.equal(with_refit.final_fit.predictions, recommended.predictions)


@pytest.mark.parametrize(
    ("representations", "position_mask", "targets", "leaf_names", "error"),
    [
        (torch.ones((6, 4), dtype=torch.float32), None, None, None, "representations"),
        (torch.ones((6, 4, 3), dtype=torch.int64), None, None, None, "representations"),
        (torch.full((6, 4, 3), float("nan")), None, None, None, "representations"),
        (None, torch.ones((6, 4), dtype=torch.float32) * 2, None, None, "position_mask"),
        (None, torch.tensor([[1.0, float("nan"), 0.0, 0.0]] * 6), None, None, "position_mask"),
        (None, torch.zeros((6, 4), dtype=torch.int64), None, None, "position_mask"),
        (None, None, torch.ones((6, 1), dtype=torch.float32), None, "targets"),
        (None, None, torch.full((6,), float("inf")), None, "targets"),
        (None, None, {"A": 1.0}, None, "Target mapping"),
        (None, None, None, ("A", "B", "C", "D", "E", "missing"), "leaf_names"),
    ],
)
def test_prepare_leaf_regression_rejects_invalid_alignment_contracts(
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
    representations,
    position_mask,
    targets,
    leaf_names,
    error,
):
    """Preparation rejects invalid shapes, values, masks, targets, and leaf permutations."""
    from phylognn import prepare_leaf_regression

    with pytest.raises((TypeError, ValueError), match=error):
        prepare_leaf_regression(
            leaf_regression_tree,
            leaf_regression_representations if representations is None else representations,
            leaf_regression_position_mask if position_mask is None else position_mask,
            leaf_regression_targets if targets is None else targets,
            leaf_names=leaf_names,
        )


def test_prepare_leaf_regression_canonicalizes_strict_numeric_inputs_and_supports_one_leaf(
    ete3_module,
):
    """Preparation accepts only numeric zero/one masks and canonicalizes accepted data."""
    from phylognn import prepare_leaf_regression

    tree = ete3_module.Tree("A;")
    data = prepare_leaf_regression(
        tree,
        torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float64),
        [[1, 0]],
        torch.tensor([3.0], dtype=torch.float64),
    )

    assert data.representations.dtype == torch.float32
    assert data.targets.dtype == torch.float32
    assert data.position_mask.dtype == torch.bool
    assert data.position_mask.tolist() == [[True, False]]


def test_prepare_leaf_regression_rejects_invalid_tree_types_and_leaf_names(
    ete3_module,
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
):
    """Tree identity and leaf names are validated before tensor preparation."""
    from phylognn import prepare_leaf_regression

    with pytest.raises(TypeError, match="tree"):
        prepare_leaf_regression(
            object(),
            leaf_regression_representations,
            leaf_regression_position_mask,
            leaf_regression_targets,
        )
    with pytest.raises(ValueError, match="unique"):
        prepare_leaf_regression(
            ete3_module.Tree("(A:1,A:1,B:1);"),
            leaf_regression_representations[:3],
            leaf_regression_position_mask[:3],
            leaf_regression_targets[:3],
        )


@pytest.mark.parametrize(
    ("config", "model_class", "model_config", "error"),
    [
        (object(), None, None, "training_config"),
        (None, _NoParameterRegressor, {}, "trainable"),
        (None, _ConstructorFailureRegressor, {}, "construction failed"),
        (None, _InvalidOutputRegressor, {"kind": "shape"}, "predictions"),
        (None, _InvalidOutputRegressor, {"kind": "nonfinite"}, "predictions"),
        (None, _InvalidOutputRegressor, {"kind": "nondifferentiable"}, "loss"),
        (None, _InvalidOutputRegressor, {"kind": "attention"}, "attention"),
        (None, _InvalidOutputRegressor, {"kind": "tuple"}, "model must return"),
    ],
)
def test_fit_leaf_regression_rejects_invalid_pre_update_contracts(
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
    config,
    model_class,
    model_config,
    error,
):
    """Invalid fit contracts fail before an invalid model can update a parameter."""
    from phylognn import LeafRegressionConfig, fit_leaf_regression, prepare_leaf_regression

    _InvalidOutputRegressor.instances.clear()
    data = prepare_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
    )
    with pytest.raises((TypeError, ValueError, RuntimeError), match=error):
        fit_leaf_regression(
            data,
            train_indices=[0],
            training_config=config or LeafRegressionConfig(epochs=1, seed=11),
            model_class=model_class,
            model_config=model_config,
        )

    assert all(instance.weight.item() == 1.0 for instance in _InvalidOutputRegressor.instances)


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"epochs": 0}, "epochs"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"weight_decay": -1.0}, "weight_decay"),
        ({"seed": True}, "seed"),
        ({"device": object()}, "device"),
    ],
)
def test_leaf_regression_config_rejects_invalid_values(kwargs, error):
    """The training config accepts only its five validated control fields."""
    from phylognn import LeafRegressionConfig

    with pytest.raises(ValueError, match=error):
        LeafRegressionConfig(**kwargs)


def test_fit_leaf_regression_rejects_invalid_indices_and_model_definitions(
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
):
    """Index and model-definition errors are rejected before model training begins."""
    from phylognn import LeafRegressionConfig, fit_leaf_regression, prepare_leaf_regression

    data = prepare_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
    )
    config = LeafRegressionConfig(epochs=1, seed=17)
    for train_indices, model_class, model_config, error in (
        ([0, 0], _PredictionOnlyRegressor, {"offset": 0.0}, "duplicate"),
        ([6], _PredictionOnlyRegressor, {"offset": 0.0}, "outside"),
        ([0.0], _PredictionOnlyRegressor, {"offset": 0.0}, "integer"),
        ([0], object, {}, "model_class"),
        ([0], _PredictionOnlyRegressor, [], "model_config"),
    ):
        with pytest.raises((TypeError, ValueError), match=error):
            fit_leaf_regression(
                data,
                train_indices=train_indices,
                training_config=config,
                model_class=model_class,
                model_config=model_config,
            )
    with pytest.raises(TypeError, match="hidden_dim"):
        LeafRegressionConfig(hidden_dim=16)


def test_cross_validation_rejects_invalid_folds_and_scores_before_fitting(
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
):
    """Fold and custom-score failures occur before any fold model is constructed."""
    from phylognn import (
        LeafRegressionConfig,
        cross_validate_leaf_regression,
        prepare_leaf_regression,
    )

    data = prepare_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
    )
    _InvalidOutputRegressor.instances.clear()
    with pytest.raises(ValueError, match="validation_folds"):
        cross_validate_leaf_regression(data, validation_folds=([0, 1], [1, 2]))
    assert not _InvalidOutputRegressor.instances
    with pytest.raises(ValueError, match="score_fn"):
        cross_validate_leaf_regression(
            data,
            validation_folds=([0, 1], [2, 3], [4, 5]),
            training_config=LeafRegressionConfig(epochs=1, seed=13),
            score_fn=lambda predictions, targets: float("nan"),
            model_class=_InvalidOutputRegressor,
            model_config={"kind": "shape"},
        )
    assert not _InvalidOutputRegressor.instances


@pytest.mark.parametrize(
    "n_splits,error",
    [(True, "integer"), (1, "2 <="), (4, "floor")],
)
def test_cross_validation_rejects_invalid_automatic_splits_before_fitting(
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
    n_splits,
    error,
):
    """Automatic split counts are validated before any model is constructed."""
    from phylognn import cross_validate_leaf_regression, prepare_leaf_regression

    data = prepare_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
    )
    _InvalidOutputRegressor.instances.clear()
    with pytest.raises((TypeError, ValueError), match=error):
        cross_validate_leaf_regression(
            data,
            n_splits=n_splits,
            model_class=_InvalidOutputRegressor,
            model_config={"kind": "shape"},
        )
    assert not _InvalidOutputRegressor.instances


def test_cross_validation_rejects_constant_default_r2_targets_before_fitting(
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_tree,
):
    """Default R-squared folds require nonconstant validation targets before fitting."""
    from phylognn import cross_validate_leaf_regression, prepare_leaf_regression

    data = prepare_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        torch.ones(6, dtype=torch.float32),
    )
    _InvalidOutputRegressor.instances.clear()
    with pytest.raises(ValueError, match="R-squared"):
        cross_validate_leaf_regression(
            data,
            validation_folds=([0, 1], [2, 3], [4, 5]),
            model_class=_InvalidOutputRegressor,
            model_config={"kind": "shape"},
        )
    assert not _InvalidOutputRegressor.instances


@pytest.mark.parametrize("entry_point", ["cross_validate", "workflow"])
def test_invalid_workflow_contracts_restore_caller_rng_state(
    entry_point,
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
):
    """CV and the facade restore Python, NumPy, and PyTorch RNG states on errors."""
    from phylognn import (
        LeafRegressionConfig,
        cross_validate_leaf_regression,
        prepare_leaf_regression,
        run_leaf_regression,
    )

    random.seed(73)
    np.random.seed(73)
    torch.manual_seed(73)
    expected = (random.random(), np.random.rand(), torch.rand(2))
    random.seed(73)
    np.random.seed(73)
    torch.manual_seed(73)

    def invalid_score(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        del predictions, targets
        random.random()
        np.random.rand()
        torch.rand(1)
        return float("nan")

    kwargs = {
        "validation_folds": ([0, 1], [2, 3], [4, 5]),
        "training_config": LeafRegressionConfig(epochs=1, seed=31),
        "score_fn": invalid_score,
    }
    with pytest.raises(ValueError, match="score_fn"):
        if entry_point == "cross_validate":
            data = prepare_leaf_regression(
                leaf_regression_tree,
                leaf_regression_representations,
                leaf_regression_position_mask,
                leaf_regression_targets,
            )
            cross_validate_leaf_regression(data, **kwargs)
        else:
            run_leaf_regression(
                leaf_regression_tree,
                leaf_regression_representations,
                leaf_regression_position_mask,
                leaf_regression_targets,
                **kwargs,
            )

    actual = (random.random(), np.random.rand(), torch.rand(2))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])
