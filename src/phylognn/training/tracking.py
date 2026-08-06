"""
Optional experiment tracking support for PhyloGNN training.

The module keeps third-party tracking dependencies isolated from the default
training import path. Importing this module never imports wandb; the wandb
package is loaded only when `WandbTracker.start()` is called for an enabled
tracking run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
import math
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Protocol, Sequence

TrackingStatus = Literal["completed", "failed", "interrupted"]
ScalarValue = str | int | float | bool | None
ConfigValue = ScalarValue | tuple[ScalarValue, ...]

# Ordered names shared by standard training and leaf-regression tracking.
FIXED_METRIC_CATALOG: tuple[str, ...] = (
    "train/loss",
    "train/lr",
    "train/epoch_time_sec",
    "val/loss",
    "final/best_val_loss",
    "final/best_epoch",
    "cv/fold_score",
    "cv/validation_leaf_count",
    "cv/mean_score",
    "cv/weighted_score",
    "cv/std_score",
    "cv/min_score",
    "cv/max_score",
    "cv/mae",
    "cv/pearson_r",
)
# Descriptive alias for callers that prefer the contract terminology.
QUANTITATIVE_METRIC_CATALOG = FIXED_METRIC_CATALOG

_OPERATIONAL_METRIC_PREFIXES = ("stage/",)
_OPERATIONAL_METRIC_KEYS = frozenset(("status/state",))

SECRET_KEY_PARTS = (
    "api_key",
    "apikey",
    "auth",
    "credential",
    "password",
    "secret",
    "token",
)


class TrackingError(RuntimeError):
    """Raised when experiment tracking cannot be configured or updated."""


@dataclass(frozen=True)
class TrackingRunInfo:
    """External tracking run identity shown in local training output."""

    run_id: Optional[str] = None
    run_url: Optional[str] = None
    run_name: Optional[str] = None


@dataclass(frozen=True)
class TrackingConfig:
    """
    Configuration for one optional experiment tracking run.

    When `enabled` is false, no external tracking dependency is imported or
    initialized. When enabled, the wandb backend validates destination and label
    metadata before training starts. ``metrics`` is an optional tuple allowlist:
    ``None`` records all applicable quantitative fields, while an empty tuple
    records none. Operational identity and lifecycle fields are unaffected.
    """

    enabled: bool = False
    backend: Literal["wandb"] = "wandb"
    project: Optional[str] = None
    entity: Optional[str] = None
    run_name: Optional[str] = None
    group: Optional[str] = None
    job_type: Optional[str] = "train"
    tags: tuple[str, ...] = field(default_factory=tuple)
    dataset_id: Optional[str] = None
    config_metadata: Mapping[str, object] = field(default_factory=dict)
    metrics: tuple[str, ...] | None = None

    def validate(self) -> None:
        """Validate tracking settings without importing optional backends."""
        if not isinstance(self.enabled, bool):
            raise TrackingError("tracking.enabled must be a boolean.")
        if self.backend != "wandb":
            raise TrackingError(f"tracking.backend must be 'wandb', got {self.backend!r}.")
        for key in ("project", "entity", "run_name", "group", "job_type", "dataset_id"):
            value = getattr(self, key)
            if value is not None and not isinstance(value, str):
                raise TrackingError(f"tracking.{key} must be a string or null.")
        if not isinstance(self.tags, tuple) or any(not isinstance(tag, str) for tag in self.tags):
            raise TrackingError("tracking.tags must contain only strings.")
        validate_metric_selection(self.metrics)
        sanitize_config_metadata(self.config_metadata)
        if self.enabled and not self.project:
            raise TrackingError("tracking.project is required when wandb tracking is enabled.")


class TrackerProtocol(Protocol):
    """Protocol used by `Trainer` for enabled tracking runs."""

    def start(self, config: Mapping[str, ConfigValue]) -> TrackingRunInfo:
        """Initialize tracking before the first training epoch."""

    def log_metrics(self, metrics: Mapping[str, float | int | str], *, step: int) -> None:
        """Log one metric payload for a completed epoch or terminal state."""

    def finish(self, status: TrackingStatus) -> None:
        """Mark the external run with its terminal status."""


class NoOpTracker:
    """Tracker used when experiment tracking is disabled."""

    def start(self, config: Mapping[str, ConfigValue]) -> TrackingRunInfo:
        return TrackingRunInfo()

    def log_metrics(self, metrics: Mapping[str, float | int | str], *, step: int) -> None:
        return None

    def finish(self, status: TrackingStatus) -> None:
        return None


class WandbTracker:
    """Weights & Biases tracker adapter with lazy optional dependency import."""

    def __init__(self, tracking_config: TrackingConfig) -> None:
        tracking_config.validate()
        self.tracking_config = tracking_config
        self._wandb: Any = None
        self._run: Any = None

    def start(self, config: Mapping[str, ConfigValue]) -> TrackingRunInfo:
        """Initialize a wandb run and return local identity metadata."""
        sanitized_config = sanitize_config_metadata(config)
        try:
            self._wandb = import_module("wandb")
        except ImportError as exc:
            raise TrackingError(
                "wandb is required when experiment tracking is enabled. "
                "Install the optional 'wandb' extra."
            ) from exc

        try:
            self._run = self._wandb.init(
                project=self.tracking_config.project,
                entity=self.tracking_config.entity,
                name=self.tracking_config.run_name,
                group=self.tracking_config.group,
                job_type=self.tracking_config.job_type,
                tags=list(self.tracking_config.tags),
                config=dict(sanitized_config),
            )
        except Exception as exc:  # pragma: no cover - exercised with fake module tests.
            raise TrackingError(f"wandb tracking could not be initialized: {exc}") from exc

        return TrackingRunInfo(
            run_id=_optional_str(getattr(self._run, "id", None)),
            run_url=_optional_str(getattr(self._run, "url", None)),
            run_name=_optional_str(getattr(self._run, "name", None)),
        )

    def log_metrics(self, metrics: Mapping[str, float | int | str], *, step: int) -> None:
        if self._run is None:
            raise TrackingError("wandb tracking has not been initialized.")
        payload = _validate_metric_payload(metrics)
        try:
            self._run.log(payload, step=step)
        except Exception as exc:  # pragma: no cover - exercised with fake module tests.
            raise TrackingError(f"wandb metrics could not be logged: {exc}") from exc

    def finish(self, status: TrackingStatus) -> None:
        if self._run is None:
            return
        try:
            finish = getattr(self._run, "finish", None)
            if callable(finish):
                finish(exit_code=0 if status == "completed" else 1)
            elif self._wandb is not None and callable(getattr(self._wandb, "finish", None)):
                self._wandb.finish(exit_code=0 if status == "completed" else 1)
        except Exception as exc:  # pragma: no cover - exercised with fake module tests.
            raise TrackingError(f"wandb run could not be finished: {exc}") from exc


def create_tracker(tracking_config: Optional[TrackingConfig]) -> TrackerProtocol:
    """Create the tracker implementation for a training run."""
    if tracking_config is None or not tracking_config.enabled:
        return NoOpTracker()
    if tracking_config.backend == "wandb":
        return WandbTracker(tracking_config)
    raise TrackingError(f"Unsupported tracking backend: {tracking_config.backend!r}.")


def sanitize_config_metadata(metadata: Mapping[str, object]) -> dict[str, ConfigValue]:
    """
    Return metadata safe for external experiment tracking.

    Secret-like keys are rejected. String values that look like local paths are
    reduced to their final file or directory name so full local paths are not
    logged by default.
    """
    if not isinstance(metadata, Mapping):
        raise TrackingError("tracking metadata must be a mapping.")

    sanitized: dict[str, ConfigValue] = {}
    for key in sorted(metadata):
        if not isinstance(key, str):
            raise TrackingError("tracking metadata keys must be strings.")
        _reject_secret_key(key)
        sanitized[key] = _sanitize_config_value(key, metadata[key])
    return sanitized


def build_experiment_config(
    *,
    model_type: str,
    model_params: Mapping[str, object],
    training_values: Mapping[str, object],
    loss_name: str,
    metric_names: Sequence[str],
    tracking_config: TrackingConfig,
    config_path: Optional[Path] = None,
) -> dict[str, ConfigValue]:
    """Build stable comparable configuration fields for tracking."""
    values: dict[str, object] = {
        "model.type": model_type,
        "loss.name": loss_name,
        "metrics.names": tuple(metric_names),
        "tracking.backend": tracking_config.backend,
        "tracking.enabled": tracking_config.enabled,
    }
    for prefix, mapping in (("model.params", model_params), ("training", training_values)):
        for key in sorted(mapping):
            values[f"{prefix}.{key}"] = mapping[key]
    if tracking_config.group is not None:
        values["tracking.group"] = tracking_config.group
    if tracking_config.job_type is not None:
        values["tracking.job_type"] = tracking_config.job_type
    if tracking_config.run_name is not None:
        values["tracking.run_name"] = tracking_config.run_name
    if tracking_config.tags:
        values["tracking.tags"] = tracking_config.tags
    if tracking_config.dataset_id is not None:
        values["data.dataset_id"] = tracking_config.dataset_id
    if config_path is not None:
        values["data.config_file"] = config_path.name
    values.update(tracking_config.config_metadata)
    return sanitize_config_metadata(values)


def build_epoch_metrics(
    *,
    train_metrics: Mapping[str, float],
    val_metrics: Optional[Mapping[str, float]],
    lr: float,
    epoch_time_sec: float,
) -> dict[str, float | int]:
    """Build an ordered, finite metric payload for one completed training epoch."""
    payload = _prefixed_metrics("train", train_metrics)
    _add_finite_metric(payload, "train/lr", lr)
    _add_finite_metric(payload, "train/epoch_time_sec", epoch_time_sec)
    if val_metrics is not None:
        payload.update(_prefixed_metrics("val", val_metrics))
    return payload


def validate_metric_selection(metrics: tuple[str, ...] | None) -> tuple[str, ...] | None:
    """Validate the structural shape and namespaces of a metric allowlist.

    ``None`` means all applicable quantitative fields and ``()`` means no
    quantitative fields. A non-empty tuple must contain unique, non-empty
    ``namespace/name`` entries. Each namespace segment is checked by the same
    sensitive-key sanitizer used for tracking metadata.
    """
    if metrics is None:
        return None
    if not isinstance(metrics, tuple):
        raise TrackingError("tracking.metrics must be a tuple of strings or null.")

    seen: set[str] = set()
    for name in metrics:
        if not isinstance(name, str):
            raise TrackingError("tracking.metrics must contain only strings.")
        if not name or name != name.strip() or name.count("/") != 1:
            raise TrackingError(
                f"tracking.metrics entry {name!r} must use a non-empty namespace/name format."
            )
        if name in seen:
            raise TrackingError(f"tracking.metrics contains duplicate name {name!r}.")
        seen.add(name)
        namespace, metric_name = name.split("/")
        if not namespace or not metric_name or any(char.isspace() for char in name):
            raise TrackingError(
                f"tracking.metrics entry {name!r} must use a non-empty namespace/name format."
            )
        _reject_secret_key(namespace)
        _reject_secret_key(metric_name)
    return metrics


def validate_workflow_metrics(
    metrics: tuple[str, ...] | None,
    available_metrics: Sequence[str] = (),
    *,
    workflow: str = "workflow",
) -> tuple[str, ...] | None:
    """Validate a selection against fixed and workflow-specific metric names."""
    validate_metric_selection(metrics)
    if metrics is None:
        return None
    available = set(FIXED_METRIC_CATALOG)
    available.update(available_metrics)
    unknown = tuple(name for name in metrics if name not in available)
    if unknown:
        names = ", ".join(repr(name) for name in unknown)
        raise TrackingError(f"tracking.metrics has unknown name(s) for {workflow}: {names}.")
    return metrics


def filter_quantitative_metrics(
    payload: Mapping[str, float | int | str],
    selection: tuple[str, ...] | None = None,
) -> dict[str, float | int | str]:
    """Filter quantitative fields while retaining operational lifecycle fields.

    Payload insertion order is preserved. ``None`` retains all quantitative
    fields, while an empty tuple removes them all.
    """
    validate_metric_selection(selection)
    selected = None if selection is None else set(selection)
    filtered: dict[str, float | int | str] = {}
    for key, value in payload.items():
        if _is_operational_metric_key(key) or selected is None or key in selected:
            filtered[key] = value
    return filtered


def build_final_metrics(
    *, best_val_loss: float, best_epoch: Optional[int]
) -> dict[str, float | int]:
    """Build final comparable metrics, omitting unavailable or non-finite values."""
    payload: dict[str, float | int] = {}
    has_best_loss = (
        not isinstance(best_val_loss, bool)
        and isinstance(best_val_loss, (int, float))
        and math.isfinite(best_val_loss)
    )
    if has_best_loss:
        payload["final/best_val_loss"] = float(best_val_loss)
    if has_best_loss and best_epoch is not None:
        _add_finite_metric(payload, "final/best_epoch", best_epoch)
    return payload


def build_status_metrics(status: TrackingStatus) -> dict[str, str]:
    """Build terminal status payload."""
    return {"status/state": status}


def _prefixed_metrics(prefix: str, metrics: Mapping[str, float]) -> dict[str, float]:
    payload: dict[str, float] = {}
    for key in sorted(metrics):
        name = "loss" if key == "loss" else str(key)
        _add_finite_metric(payload, f"{prefix}/{name}", metrics[key])
    return payload


def _add_finite_metric(payload: dict[str, float | int], key: str, value: object) -> None:
    """Add one finite scalar metric without fabricating a replacement value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return
    if math.isfinite(value):
        payload[key] = value


def _validate_metric_payload(
    metrics: Mapping[str, float | int | str],
) -> dict[str, float | int | str]:
    if not isinstance(metrics, Mapping):
        raise TrackingError("tracking metrics must be a mapping.")
    payload: dict[str, float | int | str] = {}
    for key, value in metrics.items():
        if not isinstance(key, str):
            raise TrackingError("tracking metric names must be strings.")
        if not isinstance(value, (int, float, str)) or isinstance(value, bool):
            raise TrackingError(f"tracking metric {key!r} must be numeric or string.")
        if isinstance(value, (int, float)) and not math.isfinite(value):
            raise TrackingError(f"tracking metric {key!r} must be finite.")
        payload[key] = value
    return payload


def _sanitize_config_value(key: str, value: object) -> ConfigValue:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, Path):
        return value.name
    if isinstance(value, str):
        if _looks_like_path_key(key) or "/" in value or "\\" in value:
            return Path(value).name
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_sanitize_sequence_value(key, item) for item in value)
    raise TrackingError(f"tracking metadata value for {key!r} must be scalar or a scalar list.")


def _sanitize_sequence_value(key: str, value: object) -> ScalarValue:
    sanitized = _sanitize_config_value(key, value)
    if isinstance(sanitized, tuple):
        raise TrackingError(f"tracking metadata sequence for {key!r} must not be nested.")
    return sanitized


def _reject_secret_key(key: str) -> None:
    normalized = key.lower().replace("-", "_")
    if any(part in normalized for part in SECRET_KEY_PARTS):
        raise TrackingError(f"tracking metadata key {key!r} may contain sensitive information.")


def _looks_like_path_key(key: str) -> bool:
    normalized = key.lower()
    return normalized.endswith(("path", "file", "dir")) or any(
        part in normalized for part in (".path", ".file", ".dir", "_path", "_file", "_dir")
    )


def _optional_str(value: object) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def _is_operational_metric_key(key: str) -> bool:
    return key in _OPERATIONAL_METRIC_KEYS or key.startswith(_OPERATIONAL_METRIC_PREFIXES)
