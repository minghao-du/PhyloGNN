"""TorchMetrics-backed metric construction for training workflows."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from torchmetrics import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
    Metric,
    R2Score,
)

MetricFactory = Callable[..., Metric]


def _validate_metric(metric: Metric) -> Metric:
    if not isinstance(metric, Metric):
        raise TypeError(
            "Metric factories must return torchmetrics.Metric instances, "
            f"got {type(metric).__name__}."
        )
    return metric


def _validate_num_outputs(num_outputs: int) -> int:
    if not isinstance(num_outputs, int) or isinstance(num_outputs, bool) or num_outputs <= 0:
        raise ValueError("R2 metric num_outputs must be a positive integer.")
    return num_outputs


def _mse_factory() -> Metric:
    return _validate_metric(MeanSquaredError(dist_sync_on_step=False))


def _mae_factory() -> Metric:
    return _validate_metric(MeanAbsoluteError(dist_sync_on_step=False))


def _rmse_factory() -> Metric:
    return _validate_metric(MeanSquaredError(squared=False, dist_sync_on_step=False))


def _r2_factory(*, num_outputs: int = 1) -> Metric:
    num_outputs = _validate_num_outputs(num_outputs)
    metric = R2Score(multioutput="raw_values", dist_sync_on_step=False)
    # TorchMetrics 1.x infers output dimensionality at update time. Store the
    # configured contract explicitly so Trainer can fail fast before state update.
    metric.num_outputs = num_outputs
    return _validate_metric(metric)


def _mape_factory() -> Metric:
    return _validate_metric(MeanAbsolutePercentageError(dist_sync_on_step=False))


class MetricRegistry:
    """Internal registry that resolves supported metric keys to Metric objects."""

    _FACTORIES: Mapping[str, MetricFactory] = {
        "mse": _mse_factory,
        "mae": _mae_factory,
        "rmse": _rmse_factory,
        "r2": _r2_factory,
        "mape": _mape_factory,
    }

    @classmethod
    def names(cls) -> tuple[str, ...]:
        return tuple(cls._FACTORIES)

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> Metric:
        if name not in cls._FACTORIES:
            valid = ", ".join(sorted(cls._FACTORIES))
            raise ValueError(f"Unknown metric {name!r}; expected one of ({valid}).")
        metric = cls._FACTORIES[name](**kwargs)
        return _validate_metric(metric)
