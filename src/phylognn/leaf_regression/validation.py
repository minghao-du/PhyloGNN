"""Leaf-regression validation contracts and workflow entry points."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Real
from typing import TYPE_CHECKING, TypeAlias

import torch

from phylognn.training.tracking import TrackerProtocol, TrackingConfig

from .data import LeafRegressionData, prepare_leaf_regression
from .fitting import (
    LeafFitResult,
    LeafRegressionConfig,
    _local_rng,
    _validate_indices,
    fit_leaf_regression,
)
from .tracking import _LeafExperimentCoordinator, _build_leaf_tracking_config

if TYPE_CHECKING:
    from ete3 import Tree


ScoreFunction: TypeAlias = Callable[[torch.Tensor, torch.Tensor], float | torch.Tensor]


@dataclass(frozen=True)
class LeafCrossValidationResult:
    """Detached results from one complete leaf-wise cross-validation run.

    Args:
        cv_score: Finite fold-size-weighted validation score.
        fold_scores: One finite score for each validation fold.
        oof_predictions: Finite floating out-of-fold predictions ``[N]``.
        validation_folds: Ordered, nonempty long validation-index tensors.
        fold_results: One :class:`LeafFitResult` for each validation fold.
        final_fit: Optional detached all-leaf refit result.
    """

    cv_score: float
    fold_scores: tuple[float, ...]
    oof_predictions: torch.Tensor
    validation_folds: tuple[torch.Tensor, ...]
    fold_results: tuple[LeafFitResult, ...]
    final_fit: LeafFitResult | None

    def __post_init__(self) -> None:
        cv_score = _finite_score(self.cv_score, "cv_score")
        if not isinstance(self.fold_scores, tuple) or not self.fold_scores:
            raise ValueError("`fold_scores` must be a nonempty tuple of finite scores.")
        fold_scores = tuple(_finite_score(score, "fold_scores") for score in self.fold_scores)
        oof_predictions = _detached_predictions(self.oof_predictions, "oof_predictions")
        folds = _validate_folds(self.validation_folds, oof_predictions.numel())
        if len(fold_scores) != len(folds):
            raise ValueError("`fold_scores` must contain one score per validation fold.")
        if not isinstance(self.fold_results, tuple) or len(self.fold_results) != len(folds):
            raise ValueError("`fold_results` must contain one result per validation fold.")
        if any(not isinstance(result, LeafFitResult) for result in self.fold_results):
            raise TypeError("`fold_results` must contain LeafFitResult instances.")
        if self.final_fit is not None and not isinstance(self.final_fit, LeafFitResult):
            raise TypeError("`final_fit` must be a LeafFitResult or None.")

        object.__setattr__(self, "cv_score", cv_score)
        object.__setattr__(self, "fold_scores", fold_scores)
        object.__setattr__(self, "oof_predictions", oof_predictions)
        object.__setattr__(self, "validation_folds", folds)


@dataclass(frozen=True)
class LeafRegressionResult:
    """Detached results returned by the recommended leaf-regression workflow.

    ``oof_predictions`` and ``predictions`` are finite floating tensors ``[N]``.
    Optional ``attention`` is ``[N, L]`` and ``mean_attention`` is ``[L]``.
    Attention fields must be supplied together or both be ``None``.
    """

    cv_score: float
    fold_scores: tuple[float, ...]
    oof_predictions: torch.Tensor
    predictions: torch.Tensor
    attention: torch.Tensor | None
    mean_attention: torch.Tensor | None

    def __post_init__(self) -> None:
        cv_score = _finite_score(self.cv_score, "cv_score")
        if not isinstance(self.fold_scores, tuple) or not self.fold_scores:
            raise ValueError("`fold_scores` must be a nonempty tuple of finite scores.")
        fold_scores = tuple(_finite_score(score, "fold_scores") for score in self.fold_scores)
        oof_predictions = _detached_predictions(self.oof_predictions, "oof_predictions")
        predictions = _detached_predictions(self.predictions, "predictions")
        if predictions.shape != oof_predictions.shape:
            raise ValueError("`predictions` must have shape [N] matching `oof_predictions`.")
        attention, mean_attention = _validate_attention(
            self.attention, self.mean_attention, predictions.numel()
        )

        object.__setattr__(self, "cv_score", cv_score)
        object.__setattr__(self, "fold_scores", fold_scores)
        object.__setattr__(self, "oof_predictions", oof_predictions)
        object.__setattr__(self, "predictions", predictions)
        object.__setattr__(self, "attention", attention)
        object.__setattr__(self, "mean_attention", mean_attention)


def _finite_score(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"`{field_name}` must be a finite real number.")
    return float(value)


def _detached_predictions(value: object, field_name: str) -> torch.Tensor:
    if not torch.is_tensor(value) or not value.is_floating_point():
        raise TypeError(f"`{field_name}` must be a floating torch.Tensor.")
    if value.ndim != 1 or value.numel() == 0:
        raise ValueError(f"`{field_name}` must have nonempty shape [N].")
    if not torch.isfinite(value).all():
        raise ValueError(f"`{field_name}` must contain only finite values.")
    return value.detach().clone()


def _validate_folds(value: object, leaf_count: int) -> tuple[torch.Tensor, ...]:
    if not isinstance(value, tuple) or len(value) < 2:
        raise ValueError("`validation_folds` must be a tuple containing at least two folds.")
    folds: list[torch.Tensor] = []
    for index, fold in enumerate(value):
        if not torch.is_tensor(fold) or fold.dtype != torch.long:
            raise TypeError(f"`validation_folds[{index}]` must be a torch.long Tensor.")
        if fold.ndim != 1 or fold.numel() == 0:
            raise ValueError(f"`validation_folds[{index}]` must be a nonempty index tensor.")
        if torch.any(fold < 0) or torch.any(fold >= leaf_count):
            raise ValueError(f"`validation_folds[{index}]` contains indices outside [0, N).")
        if torch.unique(fold).numel() != fold.numel():
            raise ValueError(f"`validation_folds[{index}]` must not contain duplicate indices.")
        folds.append(fold.detach().clone())
    assigned = torch.cat(folds)
    if assigned.numel() != leaf_count or torch.unique(assigned).numel() != leaf_count:
        raise ValueError("`validation_folds` must cover every leaf exactly once without overlap.")
    return tuple(folds)


def _validate_attention(
    attention: object, mean_attention: object, leaf_count: int
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if attention is None and mean_attention is None:
        return None, None
    if attention is None or mean_attention is None:
        raise ValueError(
            "`attention` and `mean_attention` must be supplied together or both be None."
        )
    if not torch.is_tensor(attention) or not attention.is_floating_point():
        raise TypeError("`attention` must be a floating torch.Tensor or None.")
    if attention.ndim != 2 or attention.size(0) != leaf_count or attention.size(1) == 0:
        raise ValueError("`attention` must have nonempty shape [N, L].")
    if not torch.isfinite(attention).all():
        raise ValueError("`attention` must contain only finite values.")
    if not torch.is_tensor(mean_attention) or not mean_attention.is_floating_point():
        raise TypeError("`mean_attention` must be a floating torch.Tensor or None.")
    if mean_attention.ndim != 1 or mean_attention.numel() != attention.size(1):
        raise ValueError("`mean_attention` must have shape [L] matching `attention`.")
    if not torch.isfinite(mean_attention).all():
        raise ValueError("`mean_attention` must contain only finite values.")
    return attention.detach().clone(), mean_attention.detach().clone()


def cross_validate_leaf_regression(
    data: LeafRegressionData,
    *,
    n_splits: int = 5,
    validation_folds: Sequence[Sequence[int] | torch.Tensor] | None = None,
    training_config: LeafRegressionConfig | None = None,
    score_fn: ScoreFunction | None = None,
    refit: bool = True,
    model_class: type[torch.nn.Module] | None = None,
    model_config: Mapping[str, object] | None = None,
    tracking_config: TrackingConfig | None = None,
    tracker: TrackerProtocol | None = None,
    _tracking_coordinator: _LeafExperimentCoordinator | None = None,
) -> LeafCrossValidationResult:
    """Cross-validate a leaf-regression model with optional scalar tracking.

    Args:
        tracking_config: Optional explicit tracking settings. Tracking remains
            disabled when omitted or disabled.
        tracker: Optional enabled-run tracker implementation for testing or a
            custom backend.
    """
    coordinator = _tracking_coordinator or _LeafExperimentCoordinator(
        tracking_config, tracker=tracker
    )
    owns_tracking = _tracking_coordinator is None
    try:
        if not isinstance(data, LeafRegressionData):
            raise TypeError("`data` must be a LeafRegressionData instance.")
        config = training_config or LeafRegressionConfig()
        if not isinstance(config, LeafRegressionConfig):
            raise TypeError("`training_config` must be a LeafRegressionConfig instance or None.")
        leaf_count = len(data.leaf_names)
        device = (
            torch.device(config.device)
            if config.device is not None
            else data.representations.device
        )
        with _local_rng(config.seed, device):
            folds = (
                _validate_manual_folds(validation_folds, leaf_count)
                if validation_folds is not None
                else _generate_folds(leaf_count, n_splits, config.seed)
            )
            _validate_scoring_contract(data.targets, folds, score_fn)
            if owns_tracking and coordinator.enabled:
                coordinator.start(
                    _build_leaf_tracking_config(
                        tracking_config=tracking_config or TrackingConfig(enabled=False),
                        workflow_type="cross_validate",
                        leaf_count=leaf_count,
                        fold_count=len(folds),
                        refit=refit,
                        training_values={
                            "epochs": config.epochs,
                            "learning_rate": config.learning_rate,
                            "weight_decay": config.weight_decay,
                            "seed": config.seed,
                        },
                        model_class=model_class,
                        model_config=model_config,
                        score_fn=score_fn,
                        device=device,
                    )
                )
            oof = torch.empty_like(data.targets)
            scores: list[float] = []
            results: list[LeafFitResult] = []
            all_indices = torch.arange(leaf_count, dtype=torch.long)
            for index, fold in enumerate(folds):
                train_indices = all_indices[~torch.isin(all_indices, fold)]
                stage_config = _stage_config(config, index)
                tracking_stage = coordinator.start_stage("cv_fold")
                result = fit_leaf_regression(
                    data,
                    train_indices=train_indices,
                    training_config=stage_config,
                    model_class=model_class,
                    model_config=model_config,
                    _tracking_coordinator=coordinator,
                    _tracking_stage=tracking_stage,
                )
                predictions = result.predictions.to(oof.device)
                oof[fold] = predictions[fold]
                fold_score = _score_fold(predictions[fold], data.targets[fold], score_fn)
                scores.append(fold_score)
                coordinator.log_summary(
                    {
                        "stage/type": "cv_fold",
                        "stage/index": index + 1,
                        "cv/fold_score": fold_score,
                        "cv/validation_leaf_count": fold.numel(),
                    }
                )
                results.append(result)
            final_fit = None
            if refit:
                tracking_stage = coordinator.start_stage("refit")
                final_fit = fit_leaf_regression(
                    data,
                    training_config=_stage_config(config, len(folds)),
                    model_class=model_class,
                    model_config=model_config,
                    _tracking_coordinator=coordinator,
                    _tracking_stage=tracking_stage,
                )
        weighted_score = (
            sum(score * fold.numel() for score, fold in zip(scores, folds, strict=True))
            / leaf_count
        )
        coordinator.log_summary({"cv/weighted_score": weighted_score})
        result = LeafCrossValidationResult(
            cv_score=weighted_score,
            fold_scores=tuple(scores),
            oof_predictions=oof,
            validation_folds=folds,
            fold_results=tuple(results),
            final_fit=final_fit,
        )
        if owns_tracking:
            coordinator.finish("completed")
        return result
    except KeyboardInterrupt:
        if owns_tracking:
            coordinator.finish_after_failure("interrupted")
        raise
    except Exception:
        if owns_tracking:
            coordinator.finish_after_failure("failed")
        raise


def run_leaf_regression(
    tree: Tree,
    representations: object,
    position_mask: object,
    targets: Mapping[str, float] | object,
    *,
    leaf_names: Sequence[str] | None = None,
    n_splits: int = 5,
    validation_folds: Sequence[Sequence[int] | torch.Tensor] | None = None,
    training_config: LeafRegressionConfig | None = None,
    score_fn: ScoreFunction | None = None,
    model_class: type[torch.nn.Module] | None = None,
    model_config: Mapping[str, object] | None = None,
    tracking_config: TrackingConfig | None = None,
    tracker: TrackerProtocol | None = None,
) -> LeafRegressionResult:
    """Run preparation, cross-validation, and final refitting in one run.

    Args:
        tracking_config: Optional explicit tracking settings. Tracking remains
            disabled when omitted or disabled.
        tracker: Optional enabled-run tracker implementation for testing or a
            custom backend.
    """
    coordinator = _LeafExperimentCoordinator(tracking_config, tracker=tracker)
    try:
        data = prepare_leaf_regression(
            tree, representations, position_mask, targets, leaf_names=leaf_names
        )
        config = training_config or LeafRegressionConfig()
        if not isinstance(config, LeafRegressionConfig):
            raise TypeError("`training_config` must be a LeafRegressionConfig instance or None.")
        if coordinator.enabled:
            fold_count = len(validation_folds) if validation_folds is not None else n_splits
            device = (
                torch.device(config.device)
                if config.device is not None
                else data.representations.device
            )
            start_config = _build_leaf_tracking_config(
                tracking_config=tracking_config or TrackingConfig(enabled=False),
                workflow_type="run",
                leaf_count=len(data.leaf_names),
                fold_count=fold_count,
                refit=True,
                training_values={
                    "epochs": config.epochs,
                    "learning_rate": config.learning_rate,
                    "weight_decay": config.weight_decay,
                    "seed": config.seed,
                },
                model_class=model_class,
                model_config=model_config,
                score_fn=score_fn,
                device=device,
            )
            coordinator.start(start_config)
        cv_result = cross_validate_leaf_regression(
            data,
            n_splits=n_splits,
            validation_folds=validation_folds,
            training_config=config,
            score_fn=score_fn,
            refit=True,
            model_class=model_class,
            model_config=model_config,
            _tracking_coordinator=coordinator,
        )
        if cv_result.final_fit is None:
            raise RuntimeError("Leaf-regression workflow requires a final refit.")
        final_fit = cv_result.final_fit
        attention = final_fit.attention
        mean_attention = attention.mean(dim=0) if attention is not None else None
        result = LeafRegressionResult(
            cv_score=cv_result.cv_score,
            fold_scores=cv_result.fold_scores,
            oof_predictions=cv_result.oof_predictions,
            predictions=final_fit.predictions,
            attention=attention,
            mean_attention=mean_attention,
        )
        coordinator.finish("completed")
        return result
    except KeyboardInterrupt:
        coordinator.finish_after_failure("interrupted")
        raise
    except Exception:
        coordinator.finish_after_failure("failed")
        raise


def _validate_manual_folds(
    value: Sequence[Sequence[int] | torch.Tensor], leaf_count: int
) -> tuple[torch.Tensor, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) < 2:
        raise ValueError("`validation_folds` must contain at least two index sequences.")
    folds = tuple(
        _validate_indices(fold, leaf_count, f"validation_folds[{index}]")
        for index, fold in enumerate(value)
    )
    assigned = torch.cat(folds)
    if assigned.numel() != leaf_count or torch.unique(assigned).numel() != leaf_count:
        raise ValueError("`validation_folds` must cover every leaf exactly once without overlap.")
    return folds


def _generate_folds(leaf_count: int, n_splits: int, seed: int | None) -> tuple[torch.Tensor, ...]:
    if isinstance(n_splits, bool) or not isinstance(n_splits, int):
        raise TypeError("`n_splits` must be an integer.")
    if n_splits < 2 or n_splits > leaf_count // 2:
        raise ValueError("`n_splits` must satisfy 2 <= n_splits <= floor(N / 2).")
    generator = torch.Generator()
    generator.manual_seed(seed if seed is not None else generator.seed())
    return tuple(torch.tensor_split(torch.randperm(leaf_count, generator=generator), n_splits))


def _validate_r2_folds(targets: torch.Tensor, folds: tuple[torch.Tensor, ...]) -> None:
    for fold in folds:
        selected = targets[fold]
        if selected.numel() < 2 or torch.sum((selected - selected.mean()) ** 2).item() == 0:
            raise ValueError(
                "Default R-squared requires nonconstant validation folds of at least two leaves."
            )


def _validate_scoring_contract(
    targets: torch.Tensor, folds: tuple[torch.Tensor, ...], score_fn: ScoreFunction | None
) -> None:
    """Reject invalid scoring contracts before any fold model is fitted."""
    if score_fn is None:
        _validate_r2_folds(targets, folds)
        return
    if not callable(score_fn):
        raise TypeError("`score_fn` must be callable or None.")
    for fold in folds:
        _score_fold(targets[fold], targets[fold], score_fn)


def _score_fold(
    predictions: torch.Tensor, targets: torch.Tensor, score_fn: ScoreFunction | None
) -> float:
    if score_fn is None:
        denominator = torch.sum((targets - targets.mean()) ** 2)
        return float((1.0 - torch.sum((predictions - targets) ** 2) / denominator).item())
    value = score_fn(predictions.detach(), targets.detach())
    if torch.is_tensor(value):
        if value.numel() != 1 or value.requires_grad or value.is_complex():
            raise ValueError("`score_fn` must return a detached real scalar.")
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError("`score_fn` must return a finite real scalar.")
    return float(value)


def _stage_config(config: LeafRegressionConfig, offset: int) -> LeafRegressionConfig:
    if config.seed is None:
        return config
    return LeafRegressionConfig(
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        seed=config.seed + offset,
        device=config.device,
    )
