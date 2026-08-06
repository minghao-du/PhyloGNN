"""Private lifecycle coordination for leaf-regression experiment tracking."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Literal
import warnings

from phylognn.training.tracking import (
    ConfigValue,
    TrackingConfig,
    TrackingError,
    TrackingRunInfo,
    TrackingStatus,
    TrackerProtocol,
    build_epoch_metrics,
    build_experiment_config,
    build_status_metrics,
    create_tracker,
    filter_quantitative_metrics,
    sanitize_config_metadata,
    validate_workflow_metrics,
)

StageType = Literal["fit", "cv_fold", "refit"]


@dataclass(frozen=True)
class _LeafTrackingStage:
    """Identity for one tracked leaf-regression training stage."""

    stage_type: StageType
    stage_index: int


class _LeafExperimentCoordinator:
    """Coordinate one optional tracker run across leaf-regression stages."""

    def __init__(
        self,
        tracking_config: TrackingConfig | None = None,
        *,
        tracker: TrackerProtocol | None = None,
    ) -> None:
        if tracking_config is not None and not isinstance(tracking_config, TrackingConfig):
            raise TypeError("`tracking_config` must be a TrackingConfig instance or None.")
        if tracking_config is not None:
            tracking_config.validate()
            if tracking_config.enabled:
                validate_workflow_metrics(
                    tracking_config.metrics,
                    workflow="leaf regression",
                )
        # An injected tracker never enables tracking by itself; only an explicit
        # enabled TrackingConfig selects a backend and starts lifecycle calls.
        self._tracking_config = (
            tracking_config if tracking_config is not None else TrackingConfig(enabled=False)
        )
        self._injected_tracker = tracker
        self._tracker: TrackerProtocol | None = None
        self._started = False
        self._terminal_status: TrackingStatus | None = None
        self._global_step = 0
        self._stage_counts: dict[StageType, int] = {"fit": 0, "cv_fold": 0, "refit": 0}
        self._stage_epochs: dict[tuple[StageType, int], int] = {}
        self._run_info = TrackingRunInfo()

    @property
    def enabled(self) -> bool:
        """Whether this coordinator will invoke a tracker."""
        return self._tracking_config.enabled

    @property
    def global_step(self) -> int:
        """Return the most recently logged global step."""
        return self._global_step

    @property
    def run_info(self) -> TrackingRunInfo:
        """Return backend identity information after a successful start."""
        return self._run_info

    @property
    def terminal_status(self) -> TrackingStatus | None:
        """Return the successfully recorded terminal status, if any."""
        return self._terminal_status

    def start(self, config: Mapping[str, object]) -> TrackingRunInfo:
        """Start the single logical run using sanitized scalar configuration."""
        if self._started:
            raise TrackingError("leaf-regression tracking has already been started.")
        if not self.enabled:
            return self._run_info
        sanitized: dict[str, ConfigValue] = sanitize_config_metadata(config)
        self._tracker = (
            self._injected_tracker
            if self._injected_tracker is not None
            else create_tracker(self._tracking_config)
        )
        try:
            self._run_info = self._tracker.start(sanitized)
        except TrackingError:
            raise
        except Exception as error:
            raise TrackingError(
                f"leaf-regression tracking could not be started: {error}"
            ) from error
        self._started = True
        identity = _format_identity(self._run_info)
        if identity:
            print(f"Tracking run: {identity}")
        return self._run_info

    def start_stage(self, stage_type: StageType) -> _LeafTrackingStage:
        """Allocate the next stable index for a fit, fold, or refit stage."""
        if stage_type not in self._stage_counts:
            raise ValueError("`stage_type` must be 'fit', 'cv_fold', or 'refit'.")
        if self.enabled and not self._started:
            raise TrackingError("leaf-regression tracking must be started before a stage.")
        self._stage_counts[stage_type] += 1
        return _LeafTrackingStage(stage_type, self._stage_counts[stage_type])

    def log_epoch(
        self,
        stage: _LeafTrackingStage,
        *,
        loss: float,
        learning_rate: float,
        epoch_time_sec: float,
    ) -> None:
        """Record one completed epoch and advance the global step."""
        if not self.enabled:
            return
        if self._terminal_status is not None:
            raise TrackingError("leaf-regression tracking is already in a terminal state.")
        _validate_finite(loss, "loss", positive=False)
        _validate_finite(learning_rate, "learning_rate", positive=True)
        _validate_finite(epoch_time_sec, "epoch_time_sec", positive=False)
        payload = build_epoch_metrics(
            train_metrics={"loss": loss},
            val_metrics=None,
            lr=learning_rate,
            epoch_time_sec=epoch_time_sec,
        )
        payload.update(
            {
                "stage/type": stage.stage_type,
                "stage/index": stage.stage_index,
                "stage/epoch": self._next_stage_epoch(stage),
            }
        )
        payload = filter_quantitative_metrics(payload, self._tracking_config.metrics)
        self._log(payload, step=self._global_step + 1)
        self._global_step += 1

    def log_summary(self, metrics: Mapping[str, float | int | str]) -> None:
        """Record a scalar summary at the most recent epoch step."""
        if not self.enabled:
            return
        if self._terminal_status is not None:
            raise TrackingError("leaf-regression tracking is already in a terminal state.")
        _validate_summary_metrics(metrics)
        payload = filter_quantitative_metrics(metrics, self._tracking_config.metrics)
        if payload:
            self._log(payload, step=self._global_step)

    def finish(self, status: TrackingStatus) -> None:
        """Record and lock one terminal state, then finish the backend run."""
        if status not in ("completed", "failed", "interrupted"):
            raise ValueError("`status` must be completed, failed, or interrupted.")
        if not self.enabled or not self._started or self._terminal_status is not None:
            return
        self._log(build_status_metrics(status), step=self._global_step)
        self._terminal_status = status
        assert self._tracker is not None
        try:
            self._tracker.finish(status)
        except TrackingError:
            raise
        except Exception as error:
            raise TrackingError(
                f"leaf-regression tracking could not be finished: {error}"
            ) from error

    def finish_after_failure(self, status: Literal["failed", "interrupted"]) -> None:
        """Best-effort failure cleanup that never replaces the primary error."""
        try:
            self.finish(status)
        except Exception as error:
            warnings.warn(
                f"Tracking cleanup failed after training {status}: {error}",
                UserWarning,
                stacklevel=2,
            )

    def _log(self, metrics: Mapping[str, float | int | str], *, step: int) -> None:
        assert self._tracker is not None
        try:
            self._tracker.log_metrics(metrics, step=step)
        except TrackingError:
            raise
        except Exception as error:
            raise TrackingError(f"leaf-regression metrics could not be logged: {error}") from error

    def _next_stage_epoch(self, stage: _LeafTrackingStage) -> int:
        key = (stage.stage_type, stage.stage_index)
        epoch = self._stage_epochs.get(key, 0) + 1
        self._stage_epochs[key] = epoch
        return epoch


def _validate_finite(value: float, field_name: str, *, positive: bool) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise TrackingError(f"tracking {field_name} must be a finite number.")
    if positive and value <= 0:
        raise TrackingError(f"tracking {field_name} must be positive.")


def _validate_summary_metrics(metrics: Mapping[str, float | int | str]) -> None:
    """Reject non-finite summary payloads before they reach a backend."""
    if not isinstance(metrics, Mapping):
        raise TrackingError("tracking summary metrics must be a mapping.")
    for key, value in metrics.items():
        if not isinstance(key, str):
            raise TrackingError("tracking summary metric names must be strings.")
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise TrackingError(f"tracking metric {key!r} must be numeric or string.")
        if isinstance(value, (int, float)) and not math.isfinite(value):
            raise TrackingError(f"tracking metric {key!r} must be finite.")


def _format_identity(info: TrackingRunInfo) -> str:
    fields = (("id", info.run_id), ("name", info.run_name), ("url", info.run_url))
    return " ".join(f"{key}={value}" for key, value in fields if value)


def _build_leaf_tracking_config(
    *,
    tracking_config: TrackingConfig,
    workflow_type: str,
    leaf_count: int,
    fold_count: int,
    refit: bool,
    training_values: Mapping[str, object],
    model_class: type[object] | None,
    model_config: Mapping[str, object] | None,
    score_fn: object,
    device: object,
) -> dict[str, ConfigValue]:
    """Build the scalar-only start payload for one leaf-regression entry point."""
    if model_config is not None and not isinstance(model_config, Mapping):
        raise TypeError("`model_config` must be a mapping or None.")
    model_params = dict(model_config) if model_config is not None else {}
    if model_class is None and "hidden_dim" not in model_params:
        model_params["hidden_dim"] = 32
    model_type = getattr(model_class, "__name__", None) or "MaskedAttentionPhyloRegressor"
    score_name = "r2" if score_fn is None else getattr(score_fn, "__name__", None) or "custom"
    config = build_experiment_config(
        model_type=model_type,
        model_params=model_params,
        training_values=training_values,
        loss_name="mse",
        metric_names=(score_name,),
        tracking_config=tracking_config,
    )
    config.update(
        {
            "workflow.type": workflow_type,
            "data.leaf_count": leaf_count,
            "cv.fold_count": fold_count,
            "cv.refit": refit,
            "cv.score": score_name,
            "training.device": str(device),
        }
    )
    return sanitize_config_metadata(config)
