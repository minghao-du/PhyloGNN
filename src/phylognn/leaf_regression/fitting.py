"""Leaf-regression fitting contracts and fitting entry point."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import copy
from contextlib import contextmanager
from dataclasses import dataclass
import inspect
import math
from numbers import Real
import random
import time
from typing import Iterator

import numpy as np
import torch

from phylognn.training.tracking import TrackerProtocol, TrackingConfig

from phylognn.training.losses import build_loss, format_loss_identifier, resolve_loss_selection

from .data import LeafRegressionData
from .tracking import _LeafExperimentCoordinator, _LeafTrackingStage, _build_leaf_tracking_config
from phylognn.models import MaskedAttentionPhyloRegressor


def _leaf_regression_config_error_factory(message: str, category: type[Exception]) -> Exception:
    return category(f"`loss`/`huber_delta`: {message}")


@dataclass(frozen=True)
class LeafRegressionConfig:
    """Validated immutable controls for one leaf-regression fit.

    Args:
        epochs: Positive number of Adam optimization steps.
        learning_rate: Finite positive Adam learning rate.
        weight_decay: Finite nonnegative Adam weight decay.
        seed: Optional non-boolean integer for local RNG isolation.
        device: Optional value accepted by :class:`torch.device`.
        loss: Supported loss identifier from the shared training loss catalog.
            Defaults to ``"mse"``, preserving existing behavior.
        huber_delta: Optional positive finite transition threshold, valid only
            when ``loss="huber"``. Omitted means unset and resolves to ``1.0``
            under Huber.
        early_stopping: Whether cross-validation folds may stop after their
            validation loss stops improving. Direct fits reject ``True``.
        early_stopping_patience: Positive number of consecutive non-improving
            held-out losses tolerated by each cross-validation fold.
    """

    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    seed: int | None = None
    device: torch.device | str | None = None
    loss: str = "mse"
    huber_delta: float | None = None
    early_stopping: bool = False
    early_stopping_patience: int = 20

    def __post_init__(self) -> None:
        if isinstance(self.epochs, bool) or not isinstance(self.epochs, int) or self.epochs <= 0:
            raise ValueError("`epochs` must be a positive integer.")
        _validate_finite_real(self.learning_rate, "learning_rate", positive=True)
        _validate_finite_real(self.weight_decay, "weight_decay", positive=False)
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise ValueError("`seed` must be a non-boolean integer or None.")
        if self.device is not None:
            try:
                torch.device(self.device)
            except (TypeError, RuntimeError) as error:
                raise ValueError("`device` must be accepted by torch.device.") from error
        if not isinstance(self.early_stopping, bool):
            raise ValueError("`early_stopping` must be a boolean.")
        if (
            isinstance(self.early_stopping_patience, bool)
            or not isinstance(self.early_stopping_patience, int)
            or self.early_stopping_patience <= 0
        ):
            raise ValueError("`early_stopping_patience` must be a positive non-boolean integer.")
        params = {} if self.huber_delta is None else {"delta": self.huber_delta}
        _, resolved_params = resolve_loss_selection(
            self.loss, params, error_factory=_leaf_regression_config_error_factory
        )
        if self.huber_delta is not None:
            object.__setattr__(self, "huber_delta", resolved_params["delta"])


@dataclass(frozen=True)
class LeafFitResult:
    """Detached results from one leaf-regression fit.

    Args:
        predictions: Finite floating all-leaf predictions. Legacy fits use
            ``[N]``; PGLS fits use ``[N, T]``.
        attention: Optional finite floating attention ``[N, L]``.
        train_indices: Nonempty unique long indices ``[K]``.
        losses: Finite Python loss values, one per optimization epoch.

    Tensor fields are detached clones at construction, so later changes to
    training graph state cannot alter this result.
    """

    predictions: torch.Tensor
    attention: torch.Tensor | None
    train_indices: torch.Tensor
    losses: tuple[float, ...]

    def __post_init__(self) -> None:
        predictions = _detached_finite_predictions(self.predictions)
        attention = None
        if self.attention is not None:
            attention = _detached_finite_float_tensor(self.attention, "attention", dimensions=2)
            if attention.size(0) != predictions.size(0):
                raise ValueError("`attention` must have shape [N, L] matching `predictions`.")
        train_indices = _detached_long_indices(self.train_indices, "train_indices")
        losses = _validate_losses(self.losses)

        object.__setattr__(self, "predictions", predictions)
        object.__setattr__(self, "attention", attention)
        object.__setattr__(self, "train_indices", train_indices)
        object.__setattr__(self, "losses", losses)


def _validate_finite_real(value: object, field_name: str, *, positive: bool) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or (value <= 0 if positive else value < 0)
    ):
        comparison = "positive" if positive else "non-negative"
        raise ValueError(f"`{field_name}` must be a finite {comparison} real number.")
    return float(value)


def _detached_finite_float_tensor(
    value: object, field_name: str, *, dimensions: int
) -> torch.Tensor:
    if not torch.is_tensor(value) or not value.is_floating_point():
        raise TypeError(f"`{field_name}` must be a floating torch.Tensor.")
    if value.ndim != dimensions or value.numel() == 0:
        raise ValueError(f"`{field_name}` must have nonempty shape with {dimensions} dimensions.")
    if not torch.isfinite(value).all():
        raise ValueError(f"`{field_name}` must contain only finite values.")
    return value.detach().clone()


def _detached_finite_predictions(value: object) -> torch.Tensor:
    if not torch.is_tensor(value) or not value.is_floating_point():
        raise TypeError("`predictions` must be a floating torch.Tensor.")
    if value.ndim not in (1, 2) or value.numel() == 0:
        raise ValueError("`predictions` must have nonempty shape [N] or [N, T].")
    if not torch.isfinite(value).all():
        raise ValueError("`predictions` must contain only finite values.")
    return value.detach().clone()


def _detached_long_indices(value: object, field_name: str) -> torch.Tensor:
    if not torch.is_tensor(value) or value.dtype != torch.long:
        raise TypeError(f"`{field_name}` must be a torch.long Tensor.")
    if value.ndim != 1 or value.numel() == 0:
        raise ValueError(f"`{field_name}` must be a nonempty one-dimensional index tensor.")
    if torch.any(value < 0) or torch.unique(value).numel() != value.numel():
        raise ValueError(f"`{field_name}` must contain unique non-negative indices.")
    return value.detach().clone()


def _validate_losses(value: object) -> tuple[float, ...]:
    if not isinstance(value, tuple):
        raise TypeError("`losses` must be a tuple of finite float values.")
    if any(
        isinstance(loss, bool) or not isinstance(loss, Real) or not math.isfinite(loss)
        for loss in value
    ):
        raise ValueError("`losses` must contain only finite real values.")
    return tuple(float(loss) for loss in value)


def fit_leaf_regression(
    data: LeafRegressionData,
    *,
    train_indices: Sequence[int] | torch.Tensor | None = None,
    training_config: LeafRegressionConfig | None = None,
    model_class: type[torch.nn.Module] | None = None,
    model_config: Mapping[str, object] | None = None,
    pgls_head: torch.nn.Module | None = None,
    pgls_loss: torch.nn.Module | None = None,
    covariances: list[torch.Tensor] | None = None,
    batch: torch.Tensor | None = None,
    tracking_config: TrackingConfig | None = None,
    tracker: TrackerProtocol | None = None,
    _tracking_coordinator: _LeafExperimentCoordinator | None = None,
    _tracking_stage: _LeafTrackingStage | None = None,
    _tracking_validation_indices: torch.Tensor | None = None,
    _tracking_score_fn: Callable[[torch.Tensor, torch.Tensor], object] | None = None,
) -> LeafFitResult:
    """Fit one fresh leaf-regression model with optional PGLS composition.

    Args:
        data: Validated leaf-regression inputs. Position masks, targets, indices,
            and Laplacian inputs are isolated from fitting mutations.
        training_config: Fit controls. With ``early_stopping=False``, this
            performs exactly ``epochs`` optimizer steps without a held-out
            validation forward. With ``early_stopping=True``, private
            cross-validation plumbing must provide held-out validation indices;
            direct fits raise :class:`ValueError` before model construction.
        model_config: Optional constructor settings for the selected model. For
            the built-in sequence regressor, ``chunk_size`` is forwarded to the
            model; omitting it retains full-batch sequence encoding. Chunked
            raw position encodings are concatenated in input leaf order before
            attention, pooling, prediction, and Laplacian work run once over
            the complete batch. Result, checkpoint, history, and tracking
            formats are unchanged.
        pgls_head: Optional final projection consuming ordered ``[N, D]`` leaf
            features and returning trait predictions ``[N, T]``.
        pgls_loss: Optional loss accepting predictions, targets, per-tree
            ``covariances``, and a leaf-to-tree ``batch`` vector.
        covariances: Optional covariance matrix list in ascending tree-ID order.
        batch: Optional int64 vector mapping each leaf to its covariance matrix.

            These four PGLS arguments must be supplied together. When omitted,
            the existing scalar model-output and configured-loss path is used.
            PGLS backbones receive the complete target-device-prepared
            ``representations`` and ``position_mask`` tensors through
            ``forward_leaf_representations`` before fit indices are selected.
        tracking_config: Optional explicit tracking settings. Tracking remains
            disabled when omitted or disabled.
        tracker: Optional enabled-run tracker implementation for testing or a
            custom backend.

    Representations already on the target device are passed as a detached
    same-storage alias to avoid an unconditional full-size clone. Fitting and
    supported models must not write in place through that alias.
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
        if config.early_stopping and _tracking_validation_indices is None:
            raise ValueError(
                "`early_stopping=True` requires held-out validation indices and is unsupported "
                "for a direct fit."
            )
        pgls_values = (pgls_head, pgls_loss, covariances, batch)
        if any(value is not None for value in pgls_values) and not all(
            value is not None for value in pgls_values
        ):
            raise ValueError(
                "`pgls_head`, `pgls_loss`, `covariances`, and `batch` must all be provided "
                "or all omitted."
            )
        pgls_enabled = all(value is not None for value in pgls_values)
        if pgls_enabled:
            if not isinstance(pgls_head, torch.nn.Module):
                raise TypeError("`pgls_head` must be a torch.nn.Module.")
            if not isinstance(pgls_loss, torch.nn.Module):
                raise TypeError("`pgls_loss` must be a torch.nn.Module.")
            loss_module = pgls_loss
            loss_identifier = "pgls"
        else:
            loss_params = {} if config.huber_delta is None else {"delta": config.huber_delta}
            loss_name, loss_params = resolve_loss_selection(config.loss, loss_params)
            loss_module = build_loss(loss_name, loss_params)
            loss_identifier = format_loss_identifier(loss_name, loss_params)
        leaf_count = len(data.leaf_names)
        indices = (
            torch.arange(leaf_count, dtype=torch.long)
            if train_indices is None
            else _validate_indices(train_indices, leaf_count, "train_indices")
        )
        device = (
            torch.device(config.device)
            if config.device is not None
            else data.representations.device
        )
        if pgls_enabled:
            _validate_pgls_metadata(covariances, batch, leaf_count, device)
        if owns_tracking and coordinator.enabled:
            coordinator.start(
                _build_leaf_tracking_config(
                    tracking_config=tracking_config or TrackingConfig(enabled=False),
                    workflow_type="fit",
                    leaf_count=leaf_count,
                    fold_count=0,
                    refit=False,
                    training_values={
                        "epochs": config.epochs,
                        "learning_rate": config.learning_rate,
                        "weight_decay": config.weight_decay,
                        "seed": config.seed,
                        "early_stopping": config.early_stopping,
                        "early_stopping_patience": config.early_stopping_patience,
                    },
                    loss_identifier=loss_identifier,
                    model_class=model_class,
                    model_config=model_config,
                    score_fn=None,
                    device=device,
                )
            )
        with _local_rng(config.seed, device):
            representations = _prepare_representations(data.representations, device)
            position_mask = data.position_mask.detach().clone().to(device)
            targets = data.targets.detach().clone().to(device)
            selected_indices = indices.to(device)
            validation_indices = (
                None
                if _tracking_validation_indices is None
                else _validate_indices(
                    _tracking_validation_indices, leaf_count, "validation_indices"
                ).to(device)
            )
            model = _construct_model(data, model_class, model_config).to(device)
            trainable_parameters = [
                parameter for parameter in model.parameters() if parameter.requires_grad
            ]
            if pgls_enabled:
                trainable_parameters.extend(
                    parameter for parameter in pgls_head.parameters() if parameter.requires_grad
                )
            if not trainable_parameters:
                raise ValueError("The model must have trainable parameters.")
            model.train()
            if pgls_enabled:
                pgls_head.train()
            predictions, _ = _forward_fit_predictions(
                model,
                representations,
                position_mask,
                leaf_count,
                representations.size(1),
                pgls_head=pgls_head if pgls_enabled else None,
            )
            loss = _fit_partition_loss(
                predictions,
                targets,
                selected_indices,
                loss_module,
                covariances=covariances if pgls_enabled else None,
                batch=batch if pgls_enabled else None,
            )
            if not loss.requires_grad or not torch.isfinite(loss):
                raise ValueError(
                    f"The selected-leaf {loss_identifier} loss must be finite and differentiable."
                )
            optimizer = torch.optim.Adam(
                trainable_parameters,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
            losses: list[float] = []
            if config.early_stopping:
                best_validation_loss = math.inf
                best_model_state: dict[str, torch.Tensor] | None = None
                best_pgls_head_state: dict[str, torch.Tensor] | None = None
                consecutive_non_improvements = 0
            tracking_stage = _tracking_stage
            if tracking_stage is None:
                tracking_stage = coordinator.start_stage("fit")
            for epoch in range(config.epochs):
                epoch_started = time.perf_counter() if coordinator.enabled else None
                optimizer.zero_grad()
                if epoch:
                    predictions, _ = _forward_fit_predictions(
                        model,
                        representations,
                        position_mask,
                        leaf_count,
                        representations.size(1),
                        pgls_head=pgls_head if pgls_enabled else None,
                    )
                    loss = _fit_partition_loss(
                        predictions,
                        targets,
                        selected_indices,
                        loss_module,
                        covariances=covariances if pgls_enabled else None,
                        batch=batch if pgls_enabled else None,
                    )
                    if not loss.requires_grad or not torch.isfinite(loss):
                        raise ValueError(
                            f"The selected-leaf {loss_identifier} loss must be "
                            "finite and differentiable."
                        )
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                tracked_validation_predictions = (
                    None if validation_indices is None else predictions[validation_indices].detach()
                )
                if config.early_stopping:
                    model.eval()
                    if pgls_enabled:
                        pgls_head.eval()
                    with torch.no_grad():
                        validation_predictions, _ = _forward_fit_predictions(
                            model,
                            representations,
                            position_mask,
                            leaf_count,
                            representations.size(1),
                            pgls_head=pgls_head if pgls_enabled else None,
                        )
                        validation_loss = _fit_partition_loss(
                            validation_predictions,
                            targets,
                            validation_indices,
                            loss_module,
                            covariances=covariances if pgls_enabled else None,
                            batch=batch if pgls_enabled else None,
                        )
                    if not torch.isfinite(validation_loss):
                        raise ValueError(
                            f"The held-out validation {loss_identifier} loss must be finite."
                        )
                    tracked_validation_predictions = validation_predictions[
                        validation_indices
                    ].detach()
                    validation_loss_value = float(validation_loss.detach().cpu())
                    if validation_loss_value < best_validation_loss:
                        best_validation_loss = validation_loss_value
                        best_model_state = copy.deepcopy(model.state_dict())
                        if pgls_enabled:
                            best_pgls_head_state = copy.deepcopy(pgls_head.state_dict())
                        consecutive_non_improvements = 0
                    else:
                        consecutive_non_improvements += 1
                    model.train()
                    if pgls_enabled:
                        pgls_head.train()
                coordinator.log_epoch(
                    tracking_stage,
                    train_predictions=predictions[selected_indices].detach(),
                    train_targets=targets[selected_indices].detach(),
                    val_predictions=tracked_validation_predictions,
                    val_targets=(
                        None if validation_indices is None else targets[validation_indices].detach()
                    ),
                    score_fn=_tracking_score_fn,
                    loss_fn=loss_module,
                    learning_rate=float(optimizer.param_groups[0]["lr"]),
                    epoch_time_sec=(
                        time.perf_counter() - epoch_started if epoch_started is not None else 0.0
                    ),
                )
                if (
                    config.early_stopping
                    and consecutive_non_improvements >= config.early_stopping_patience
                ):
                    break
            if config.early_stopping:
                if best_model_state is None:
                    raise RuntimeError("Early stopping did not observe a held-out validation loss.")
                model.load_state_dict(best_model_state)
                if pgls_enabled:
                    if best_pgls_head_state is None:
                        raise RuntimeError("Early stopping did not retain PGLS head state.")
                    pgls_head.load_state_dict(best_pgls_head_state)
            model.eval()
            if pgls_enabled:
                pgls_head.eval()
            with torch.no_grad():
                predictions, attention = _forward_fit_predictions(
                    model,
                    representations,
                    position_mask,
                    leaf_count,
                    representations.size(1),
                    pgls_head=pgls_head if pgls_enabled else None,
                )
        result = LeafFitResult(
            predictions=predictions,
            attention=attention,
            train_indices=indices,
            losses=tuple(losses),
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


def _prepare_representations(representations: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Detach local representations or transfer them to the fitting device."""
    if representations.device == device:
        return representations.detach()
    return representations.to(device)


def _validate_indices(
    value: Sequence[int] | torch.Tensor, leaf_count: int, field_name: str
) -> torch.Tensor:
    try:
        indices = torch.as_tensor(value)
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"`{field_name}` must be a one-dimensional integer index sequence."
        ) from error
    if indices.ndim != 1 or indices.numel() == 0:
        raise ValueError(f"`{field_name}` must be a nonempty one-dimensional index sequence.")
    if indices.dtype == torch.bool or indices.is_floating_point() or indices.is_complex():
        raise TypeError(f"`{field_name}` must contain integer indices.")
    indices = indices.to(dtype=torch.long)
    if torch.any(indices < 0) or torch.any(indices >= leaf_count):
        raise ValueError(f"`{field_name}` contains indices outside [0, N).")
    if torch.unique(indices).numel() != indices.numel():
        raise ValueError(f"`{field_name}` must not contain duplicate indices.")
    return indices.cpu()


def _construct_model(
    data: LeafRegressionData,
    model_class: type[torch.nn.Module] | None,
    model_config: Mapping[str, object] | None,
) -> torch.nn.Module:
    if model_config is not None and not isinstance(model_config, Mapping):
        raise TypeError("`model_config` must be a mapping or None.")
    options = dict(model_config) if model_config is not None else {}
    if model_class is None:
        if "input_dim" in options or "leaf_laplacian" in options:
            raise ValueError("Default-model `model_config` cannot override injected fields.")
        return MaskedAttentionPhyloRegressor(
            input_dim=data.representations.size(-1),
            hidden_dim=options.pop("hidden_dim", 32),
            leaf_laplacian=data.leaf_laplacian.detach().clone(),
            chunk_size=options.pop("chunk_size", None),
            **options,
        )
    if not isinstance(model_class, type) or not issubclass(model_class, torch.nn.Module):
        raise TypeError("`model_class` must be a torch.nn.Module subclass or None.")
    return model_class(**options)


def _forward_fit_predictions(
    model: torch.nn.Module,
    representations: torch.Tensor,
    position_mask: torch.Tensor,
    leaf_count: int,
    position_count: int,
    *,
    pgls_head: torch.nn.Module | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run either the legacy prediction path or the explicit PGLS composition."""
    if pgls_head is None:
        return _normalize_model_output(
            model(representations, position_mask),
            leaf_count,
            position_count,
            position_mask,
        )

    representation_forward = getattr(model, "forward_leaf_representations", None)
    if not callable(representation_forward):
        raise TypeError(
            "PGLS backbones must define callable "
            "`forward_leaf_representations(representations, position_mask)`."
        )
    try:
        inspect.signature(representation_forward).bind(representations, position_mask)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "PGLS backbone has an incompatible "
            "`forward_leaf_representations(representations, position_mask)` contract."
        ) from error
    leaf_features = representation_forward(representations, position_mask)
    if not torch.is_tensor(leaf_features) or not leaf_features.is_floating_point():
        raise TypeError("PGLS leaf representations must be a floating torch.Tensor.")
    if leaf_features.ndim != 2 or leaf_features.shape[0] != leaf_count:
        raise ValueError("PGLS leaf representations must have ordered shape [N, D].")
    if leaf_features.shape[1] == 0 or not torch.isfinite(leaf_features).all():
        raise ValueError("PGLS leaf representations must be nonempty and finite.")
    predictions = pgls_head(leaf_features)
    if not torch.is_tensor(predictions) or not predictions.is_floating_point():
        raise TypeError("PGLS predictions must be a floating torch.Tensor.")
    if predictions.ndim != 2 or predictions.shape[0] != leaf_count or predictions.shape[1] == 0:
        raise ValueError("PGLS predictions must have nonempty shape [N, T].")
    if not torch.isfinite(predictions).all():
        raise ValueError("PGLS predictions must contain only finite values.")
    return predictions, None


def _fit_partition_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    indices: torch.Tensor,
    loss_module: torch.nn.Module,
    *,
    covariances: list[torch.Tensor] | None,
    batch: torch.Tensor | None,
) -> torch.Tensor:
    """Compute the legacy loss or a deterministically subset PGLS objective."""
    if covariances is None or batch is None:
        return loss_module(predictions[indices], targets[indices])
    ordered_indices, subset_covariances, subset_batch = _subset_pgls_metadata(
        indices, covariances, batch
    )
    prediction_indices = ordered_indices.to(device=predictions.device)
    target_indices = ordered_indices.to(device=targets.device)
    return loss_module(
        predictions.index_select(0, prediction_indices),
        targets.index_select(0, target_indices),
        subset_covariances,
        subset_batch,
    )


def _subset_pgls_metadata(
    indices: torch.Tensor,
    covariances: list[torch.Tensor],
    batch: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
    """Subset leaves stably and compact retained tree IDs in ascending order."""
    selected = torch.zeros(batch.shape[0], dtype=torch.bool, device=batch.device)
    selected[indices.to(device=batch.device)] = True
    ordered_indices: list[torch.Tensor] = []
    subset_covariances: list[torch.Tensor] = []
    subset_batches: list[torch.Tensor] = []
    compact_tree_id = 0
    for tree_id, covariance in enumerate(covariances):
        tree_indices = torch.where(batch == tree_id)[0]
        retained_positions = torch.where(selected.index_select(0, tree_indices))[0]
        if retained_positions.numel() == 0:
            continue
        retained_indices = tree_indices.index_select(0, retained_positions)
        covariance_positions = retained_positions.to(device=covariance.device)
        subset_covariance = covariance.index_select(0, covariance_positions).index_select(
            1, covariance_positions
        )
        ordered_indices.append(retained_indices)
        subset_covariances.append(subset_covariance)
        subset_batches.append(
            torch.full(
                (retained_indices.numel(),),
                compact_tree_id,
                dtype=torch.long,
                device=batch.device,
            )
        )
        compact_tree_id += 1
    if not ordered_indices:
        raise ValueError("The selected PGLS partition must contain at least one represented tree.")
    return (
        torch.cat(ordered_indices),
        subset_covariances,
        torch.cat(subset_batches),
    )


def _validate_pgls_metadata(
    covariances: object,
    batch: object,
    leaf_count: int,
    device: torch.device,
) -> None:
    """Validate full-batch metadata before deterministic partition subsetting."""
    if not isinstance(covariances, list):
        raise TypeError("`covariances` must be a list of torch.Tensor objects.")
    if not torch.is_tensor(batch):
        raise TypeError("`batch` must be a torch.Tensor.")
    if not covariances:
        raise ValueError("`covariances` must be a non-empty list.")
    if batch.ndim != 1 or batch.shape[0] != leaf_count:
        raise ValueError("`batch` must have shape [N] matching the prepared leaf count.")
    if batch.dtype != torch.long:
        raise ValueError("`batch` dtype must be torch.int64.")
    if batch.device != device:
        raise ValueError("`batch` device must match the fitting device.")

    represented_trees = torch.unique(batch, sorted=True)
    if represented_trees[0].item() < 0:
        raise ValueError("`batch` identifiers must be non-negative.")
    expected_trees = torch.arange(represented_trees.numel(), dtype=torch.long, device=batch.device)
    if not torch.equal(represented_trees, expected_trees):
        raise ValueError("`batch` identifiers must be contiguous and cover 0..K-1.")
    if represented_trees.numel() != len(covariances):
        raise ValueError("The covariance count must match the represented batch tree count.")

    for tree_id, covariance in enumerate(covariances):
        if not torch.is_tensor(covariance):
            raise TypeError("Each covariance must be a torch.Tensor.")
        if covariance.device != device:
            raise ValueError("Each covariance device must match the fitting device.")
        tree_leaf_count = torch.count_nonzero(batch == tree_id).item()
        if covariance.ndim != 2 or covariance.shape != (tree_leaf_count, tree_leaf_count):
            raise ValueError(
                f"Covariance {tree_id} shape must match its tree leaf count "
                f"({tree_leaf_count})."
            )


def _normalize_model_output(
    output: object,
    leaf_count: int,
    position_count: int,
    position_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    prediction_only = torch.is_tensor(output)
    if prediction_only:
        predictions = output
        attention = None
    elif isinstance(output, tuple) and len(output) == 2:
        predictions, attention = output
    else:
        raise TypeError(
            "The model must return finite predictions or a `(predictions, attention)` tuple."
        )
    if not torch.is_tensor(predictions) or not predictions.is_floating_point():
        raise TypeError("Model predictions must be a floating torch.Tensor.")
    if predictions.shape != (leaf_count,) or not torch.isfinite(predictions).all():
        raise ValueError("Model predictions must be finite with shape [N].")
    if prediction_only:
        return predictions, None
    if attention is None:
        raise TypeError("Model attention must be a floating torch.Tensor in the output pair.")
    if not torch.is_tensor(attention) or not attention.is_floating_point():
        raise TypeError("Model attention must be a floating torch.Tensor.")
    if attention.shape != (leaf_count, position_count) or not torch.isfinite(attention).all():
        raise ValueError("Model attention must be finite with shape [N, L].")
    masked_attention = attention.detach() * position_mask.to(dtype=attention.dtype)
    return predictions, masked_attention


@contextmanager
def _local_rng(seed: int | None, device: torch.device) -> Iterator[None]:
    cuda_devices = []
    if device.type == "cuda" and torch.cuda.is_available():
        cuda_devices.append(
            device.index if device.index is not None else torch.cuda.current_device()
        )
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    with torch.random.fork_rng(devices=cuda_devices):
        try:
            if seed is not None:
                torch.manual_seed(seed)
                random.seed(seed)
                np.random.seed(seed % (2**32))
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
