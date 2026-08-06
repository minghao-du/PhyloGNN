"""Leaf-regression fitting contracts and fitting entry point."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import math
from numbers import Real
import random
import time
from typing import Iterator

import numpy as np
import torch

from phylognn.training.tracking import TrackerProtocol, TrackingConfig

from .data import LeafRegressionData
from .tracking import _LeafExperimentCoordinator, _LeafTrackingStage, _build_leaf_tracking_config
from phylognn.models import MaskedAttentionPhyloRegressor


@dataclass(frozen=True)
class LeafRegressionConfig:
    """Validated immutable controls for one leaf-regression fit.

    Args:
        epochs: Positive number of Adam optimization steps.
        learning_rate: Finite positive Adam learning rate.
        weight_decay: Finite nonnegative Adam weight decay.
        seed: Optional non-boolean integer for local RNG isolation.
        device: Optional value accepted by :class:`torch.device`.
    """

    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    seed: int | None = None
    device: torch.device | str | None = None

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


@dataclass(frozen=True)
class LeafFitResult:
    """Detached results from one leaf-regression fit.

    Args:
        predictions: Finite floating all-leaf predictions ``[N]``.
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
        predictions = _detached_finite_float_tensor(self.predictions, "predictions", dimensions=1)
        attention = None
        if self.attention is not None:
            attention = _detached_finite_float_tensor(self.attention, "attention", dimensions=2)
            if attention.size(0) != predictions.numel():
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
    tracking_config: TrackingConfig | None = None,
    tracker: TrackerProtocol | None = None,
    _tracking_coordinator: _LeafExperimentCoordinator | None = None,
    _tracking_stage: _LeafTrackingStage | None = None,
) -> LeafFitResult:
    """Fit one fresh leaf-regression model with optional scalar tracking.

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
                    },
                    model_class=model_class,
                    model_config=model_config,
                    score_fn=None,
                    device=device,
                )
            )
        with _local_rng(config.seed, device):
            representations = data.representations.detach().clone().to(device)
            position_mask = data.position_mask.detach().clone().to(device)
            targets = data.targets.detach().clone().to(device)
            selected_indices = indices.to(device)
            model = _construct_model(data, model_class, model_config).to(device)
            trainable_parameters = [
                parameter for parameter in model.parameters() if parameter.requires_grad
            ]
            if not trainable_parameters:
                raise ValueError("The model must have trainable parameters.")
            model.train()
            predictions, _ = _normalize_model_output(
                model(representations, position_mask),
                leaf_count,
                representations.size(1),
                position_mask,
            )
            loss = torch.nn.functional.mse_loss(
                predictions[selected_indices], targets[selected_indices]
            )
            if not loss.requires_grad or not torch.isfinite(loss):
                raise ValueError("The selected-leaf MSE loss must be finite and differentiable.")
            optimizer = torch.optim.Adam(
                trainable_parameters,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
            losses: list[float] = []
            tracking_stage = _tracking_stage
            if tracking_stage is None:
                tracking_stage = coordinator.start_stage("fit")
            for epoch in range(config.epochs):
                epoch_started = time.perf_counter() if coordinator.enabled else None
                optimizer.zero_grad()
                if epoch:
                    predictions, _ = _normalize_model_output(
                        model(representations, position_mask),
                        leaf_count,
                        representations.size(1),
                        position_mask,
                    )
                    loss = torch.nn.functional.mse_loss(
                        predictions[selected_indices], targets[selected_indices]
                    )
                    if not loss.requires_grad or not torch.isfinite(loss):
                        raise ValueError(
                            "The selected-leaf MSE loss must be finite and differentiable."
                        )
                loss.backward()
                optimizer.step()
                loss_value = float(loss.detach().cpu())
                losses.append(loss_value)
                coordinator.log_epoch(
                    tracking_stage,
                    loss=loss_value,
                    learning_rate=float(optimizer.param_groups[0]["lr"]),
                    epoch_time_sec=(
                        time.perf_counter() - epoch_started if epoch_started is not None else 0.0
                    ),
                )
            model.eval()
            with torch.no_grad():
                predictions, attention = _normalize_model_output(
                    model(representations, position_mask),
                    leaf_count,
                    representations.size(1),
                    position_mask,
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
            **options,
        )
    if not isinstance(model_class, type) or not issubclass(model_class, torch.nn.Module):
        raise TypeError("`model_class` must be a torch.nn.Module subclass or None.")
    return model_class(**options)


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
