"""
TOML-backed training setup utilities.

This module parses local TOML files with Python's standard `tomllib` reader and
turns model/training configuration into the existing PhyloGNN objects. It only
configures model and training setup; datasets and loaders stay supplied by the
caller through the existing `Trainer.fit()` API.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - project runtime is Python >=3.12.
    tomllib = None

import torch.nn as nn

from phylognn.models import GATBiLSTMNet
from phylognn.training.losses import build_loss, format_loss_identifier, resolve_loss_selection
from phylognn.training.metrics import MetricRegistry
from phylognn.training.trainer import LossFn, MetricsMap, Trainer, TrainingConfig
from phylognn.training.tracking import (
    TrackingConfig,
    TrackingError,
    build_experiment_config,
    validate_workflow_metrics,
)


class TrainingConfigError(ValueError):
    """Raised when a TOML training configuration cannot be parsed or validated."""


@dataclass(frozen=True)
class ConfiguredTrainingSetup:
    """
    Effective setup produced from a TOML training configuration.

    Attributes:
        model: Configured `nn.Module` instance. The model keeps its existing
            graph contract: `data.x`, `data.edge_index`, `data.batch`, and
            `data.time_bin` when temporal mode is not `none`.
        training_config: Existing `TrainingConfig` built from effective values.
        loss_fn: Selected built-in loss callable.
        metrics: Selected built-in metrics keyed by metric name.
        tracking_config: Optional experiment tracking configuration resolved
            from `[tracking]`. Missing sections default to disabled tracking.
        tracking_metadata: Sanitized comparable run metadata passed to
            `Trainer` when tracking is enabled.
    """

    model: nn.Module
    training_config: TrainingConfig
    loss_fn: LossFn
    metrics: MetricsMap
    tracking_config: TrackingConfig
    tracking_metadata: Mapping[str, object]


PathLike = Union[str, Path]

SUPPORTED_MODEL_TYPES = frozenset({"GATBiLSTMNet", "GATNodeRegressor"})

TOP_LEVEL_KEYS = frozenset({"model", "training", "loss", "metrics", "tracking"})
MODEL_KEYS = frozenset({"type", "params"})
MODEL_PARAM_KEYS = frozenset(
    {
        "input_dim",
        "output_dim",
        "preprocess_dim",
        "gat_hidden_dim",
        "gat_heads",
        "num_gat_layers",
        "dropout_prob",
        "use_preprocessing",
        "encoder_type",
        "temporal_mode",
        "num_time_bins",
        "temporal_hidden_dim",
        "temporal_fc_hidden_dims",
        "num_lstm_layers",
        "temporal_aggregation",
        "graph_pool",
        "head_hidden_dim",
        "output_positive",
    }
)
GAT_NODE_PARAM_KEYS = frozenset(
    {
        "input_dim",
        "output_dim",
        "preprocess_dim",
        "gat_hidden_dim",
        "gat_heads",
        "num_gat_layers",
        "dropout_prob",
        "use_preprocessing",
        "encoder_type",
        "head_hidden_dim",
    }
)
REQUIRED_MODEL_PARAM_KEYS = frozenset({"input_dim", "output_dim"})
TRAINING_KEYS = frozenset(TrainingConfig.__dataclass_fields__.keys())
TRAINING_INT_KEYS = frozenset({"epochs", "batch_size", "scheduler_patience", "num_workers"})
TRAINING_FLOAT_KEYS = frozenset(
    {"learning_rate", "weight_decay", "scheduler_factor", "gradient_clip_val"}
)
TRAINING_OPTIONAL_INT_KEYS = frozenset({"early_stopping_patience"})
TRAINING_STR_KEYS = frozenset({"optimizer", "scheduler", "device", "save_dir"})
TRAINING_BOOL_KEYS = frozenset(
    {"save_best_only", "verbose", "pin_memory", "train_shuffle", "non_blocking"}
)
LOSS_KEYS = frozenset({"name", "params"})
METRICS_KEYS = frozenset({"names"})
TRACKING_KEYS = frozenset(
    {
        "enabled",
        "backend",
        "project",
        "entity",
        "run_name",
        "group",
        "job_type",
        "tags",
        "dataset_id",
        "metrics",
    }
)

METRIC_REGISTRY = frozenset(MetricRegistry.names())


def load_training_config(
    path: PathLike,
    *,
    model_overrides: Optional[Mapping[str, object]] = None,
    training_overrides: Optional[Mapping[str, object]] = None,
    loss: Optional[str] = None,
    loss_params: Optional[Mapping[str, object]] = None,
    metrics: Optional[Sequence[str]] = None,
) -> ConfiguredTrainingSetup:
    """
    Load a TOML file and create model/training setup objects.

    The TOML file must contain `[model]`, `[model.params]`, and `[training]`
    sections. It may also contain `[loss]`, `[loss.params]`, and `[metrics]`.
    Explicit keyword overrides are applied after TOML values and must satisfy
    the same validation rules, including model counters being positive
    non-bool Python integers. Supplying `loss` replaces the entire `[loss]`
    section (name and params together); `loss_params` alone overrides only the
    file's params while keeping its name. Direct constructor `TypeError` and
    `ValueError` failures are wrapped as `TrainingConfigError` with path
    context. Dataset construction, splits, and data loaders are intentionally
    out of scope for this helper.

    Raises:
        TrainingConfigError: If the file cannot be read, TOML is malformed,
            sections or keys are invalid, or object construction validation
            fails.
    """
    config_path = Path(path)
    raw_config = _load_toml(config_path)
    _validate_config_document(raw_config, config_path=config_path)

    model_config = _require_mapping(raw_config["model"], "model", config_path)
    model_params = _copy_mapping(model_config["params"], "model.params", config_path)
    training_values = _copy_mapping(raw_config["training"], "training", config_path)

    model_type = model_config["type"]
    param_keys = GAT_NODE_PARAM_KEYS if model_type == "GATNodeRegressor" else MODEL_PARAM_KEYS
    model_params = _merge_overrides(
        model_params,
        model_overrides,
        section="model.params",
        allowed_keys=param_keys,
        config_path=config_path,
    )
    training_values = _merge_overrides(
        training_values,
        training_overrides,
        section="training",
        allowed_keys=TRAINING_KEYS,
        config_path=config_path,
    )

    loss_name, loss_params_resolved = _resolve_loss_section(
        raw_config, loss, loss_params, config_path=config_path
    )
    loss_identifier = format_loss_identifier(loss_name, loss_params_resolved)
    metric_names = _resolve_metric_names(raw_config, metrics, config_path=config_path)

    _validate_required_model_params(model_params, config_path=config_path)
    _validate_training_values(training_values, config_path=config_path)

    try:
        model_type = model_config["type"]
        if model_type == "GATNodeRegressor":
            from phylognn.models import GATNodeRegressor

            model = GATNodeRegressor(**model_params)
        else:
            model = GATBiLSTMNet(**model_params)
        training_config = TrainingConfig(**training_values)
        training_config.validate()
        tracking_config = _resolve_tracking_config(
            raw_config,
            metric_names=metric_names,
            config_path=config_path,
        )
        tracking_metadata = build_experiment_config(
            model_type=model_config["type"],
            model_params=model_params,
            training_values=asdict(training_config),
            loss_name=loss_identifier,
            metric_names=metric_names,
            tracking_config=tracking_config,
            config_path=config_path,
        )
    except (TypeError, ValueError, TrackingError) as exc:
        raise TrainingConfigError(f"{config_path}: invalid configuration value: {exc}") from exc

    return ConfiguredTrainingSetup(
        model=model,
        training_config=training_config,
        loss_fn=build_loss(loss_name, loss_params_resolved),
        metrics=_resolve_metric_specs(metric_names, output_dim=model_params.get("output_dim")),
        tracking_config=tracking_config,
        tracking_metadata=tracking_metadata,
    )


def create_trainer_from_config(
    path: PathLike,
    *,
    model_overrides: Optional[Mapping[str, object]] = None,
    training_overrides: Optional[Mapping[str, object]] = None,
    loss: Optional[str] = None,
    loss_params: Optional[Mapping[str, object]] = None,
    metrics: Optional[Sequence[str]] = None,
) -> Trainer:
    """
    Create a `Trainer` from a TOML training configuration.

    This function creates the model, `TrainingConfig`, loss, and metrics, then
    delegates to the existing `Trainer`. `loss` and `loss_params` are forwarded
    unchanged to `load_training_config`. Data remains caller-provided through
    `Trainer.fit(train_loader=...)`, `Trainer.fit(train_dataset=...)`, or the
    corresponding validation arguments.
    """
    setup = load_training_config(
        path,
        model_overrides=model_overrides,
        training_overrides=training_overrides,
        loss=loss,
        loss_params=loss_params,
        metrics=metrics,
    )
    return Trainer(
        model=setup.model,
        config=setup.training_config,
        loss_fn=setup.loss_fn,
        metrics=setup.metrics,
        tracking_config=setup.tracking_config,
        tracking_metadata=setup.tracking_metadata,
    )


def _load_toml(config_path: Path) -> Mapping[str, Any]:
    if tomllib is None:
        raise TrainingConfigError(
            "Python stdlib tomllib is required to load TOML training configurations."
        )

    try:
        with config_path.open("rb") as config_file:
            data = tomllib.load(config_file)
    except OSError as exc:
        raise TrainingConfigError(f"{config_path}: cannot read TOML configuration: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise TrainingConfigError(f"{config_path}: invalid TOML: {exc}") from exc

    if not isinstance(data, dict):
        raise TrainingConfigError(f"{config_path}: TOML document must be a table.")
    return data


def _validate_config_document(raw_config: Mapping[str, Any], *, config_path: Path) -> None:
    _reject_unknown_keys(raw_config, TOP_LEVEL_KEYS, section="<root>", config_path=config_path)

    model_config = _require_section(raw_config, "model", config_path)
    training_config = _require_section(raw_config, "training", config_path)

    _reject_unknown_keys(model_config, MODEL_KEYS, section="model", config_path=config_path)
    if "type" not in model_config:
        _raise_missing("model.type", config_path)
    if model_config["type"] not in SUPPORTED_MODEL_TYPES:
        supported = ", ".join(sorted(SUPPORTED_MODEL_TYPES))
        raise TrainingConfigError(
            f"{config_path}: model.type must be one of ({supported}), got {model_config['type']!r}."
        )
    if "params" not in model_config:
        _raise_missing("model.params", config_path)

    model_type = model_config["type"]
    param_keys = GAT_NODE_PARAM_KEYS if model_type == "GATNodeRegressor" else MODEL_PARAM_KEYS
    model_params = _require_mapping(model_config["params"], "model.params", config_path)
    _reject_unknown_keys(model_params, param_keys, section="model.params", config_path=config_path)

    _reject_unknown_keys(
        training_config, TRAINING_KEYS, section="training", config_path=config_path
    )

    if "loss" in raw_config:
        loss_config = _require_mapping(raw_config["loss"], "loss", config_path)
        _reject_unknown_keys(loss_config, LOSS_KEYS, section="loss", config_path=config_path)

    if "metrics" in raw_config:
        metrics_config = _require_mapping(raw_config["metrics"], "metrics", config_path)
        _reject_unknown_keys(
            metrics_config, METRICS_KEYS, section="metrics", config_path=config_path
        )

    if "tracking" in raw_config:
        tracking_config = _require_mapping(raw_config["tracking"], "tracking", config_path)
        _reject_unknown_keys(
            tracking_config, TRACKING_KEYS, section="tracking", config_path=config_path
        )


def _require_section(
    raw_config: Mapping[str, Any], section: str, config_path: Path
) -> Mapping[str, Any]:
    if section not in raw_config:
        _raise_missing(section, config_path)
    return _require_mapping(raw_config[section], section, config_path)


def _require_mapping(value: Any, section: str, config_path: Path) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise TrainingConfigError(f"{config_path}: section [{section}] must be a TOML table.")
    return value


def _copy_mapping(value: Any, section: str, config_path: Path) -> dict[str, Any]:
    return dict(deepcopy(_require_mapping(value, section, config_path)))


def _reject_unknown_keys(
    values: Mapping[str, Any],
    allowed_keys: frozenset[str],
    *,
    section: str,
    config_path: Path,
) -> None:
    unknown_keys = sorted(set(values) - allowed_keys)
    if unknown_keys:
        joined = ", ".join(repr(key) for key in unknown_keys)
        raise TrainingConfigError(f"{config_path}: unknown key(s) in [{section}]: {joined}.")


def _merge_overrides(
    base_values: Mapping[str, Any],
    overrides: Optional[Mapping[str, object]],
    *,
    section: str,
    allowed_keys: frozenset[str],
    config_path: Path,
) -> dict[str, Any]:
    merged = dict(deepcopy(base_values))
    if overrides is None:
        return merged
    if not isinstance(overrides, Mapping):
        raise TrainingConfigError(f"{config_path}: overrides for [{section}] must be a mapping.")

    _reject_unknown_keys(overrides, allowed_keys, section=section, config_path=config_path)
    merged.update(deepcopy(dict(overrides)))
    return merged


def _validate_required_model_params(model_params: Mapping[str, Any], *, config_path: Path) -> None:
    for key in sorted(REQUIRED_MODEL_PARAM_KEYS):
        if key not in model_params:
            _raise_missing(f"model.params.{key}", config_path)


def _validate_training_values(training_values: dict[str, Any], *, config_path: Path) -> None:
    if "scheduler" in training_values and training_values["scheduler"] == "none":
        training_values["scheduler"] = None
    if "early_stopping_patience" not in training_values:
        training_values["early_stopping_patience"] = None
    elif training_values["early_stopping_patience"] in {"none", 0, -1}:
        training_values["early_stopping_patience"] = None

    for key in sorted(TRAINING_INT_KEYS):
        if key in training_values and (
            not isinstance(training_values[key], int) or isinstance(training_values[key], bool)
        ):
            _raise_type_error(f"training.{key}", "an integer", training_values[key], config_path)

    for key in sorted(TRAINING_FLOAT_KEYS):
        if (
            key in training_values
            and training_values[key] is not None
            and (
                not isinstance(training_values[key], (int, float))
                or isinstance(training_values[key], bool)
            )
        ):
            _raise_type_error(f"training.{key}", "a number", training_values[key], config_path)

    for key in sorted(TRAINING_OPTIONAL_INT_KEYS):
        if (
            key in training_values
            and training_values[key] is not None
            and (
                not isinstance(training_values[key], int) or isinstance(training_values[key], bool)
            )
        ):
            _raise_type_error(
                f"training.{key}", "an integer, 'none', or null", training_values[key], config_path
            )

    for key in sorted(TRAINING_STR_KEYS):
        if (
            key in training_values
            and training_values[key] is not None
            and not isinstance(training_values[key], str)
        ):
            _raise_type_error(f"training.{key}", "a string", training_values[key], config_path)

    for key in sorted(TRAINING_BOOL_KEYS):
        if key in training_values and not isinstance(training_values[key], bool):
            _raise_type_error(f"training.{key}", "a boolean", training_values[key], config_path)


def _resolve_loss_section(
    raw_config: Mapping[str, Any],
    explicit_loss: Optional[str],
    explicit_loss_params: Optional[Mapping[str, object]],
    *,
    config_path: Path,
) -> tuple[str, dict[str, float]]:
    if explicit_loss_params is not None and not isinstance(explicit_loss_params, Mapping):
        raise TrainingConfigError(f"{config_path}: `loss_params` must be a mapping or None.")

    loss_config = _require_mapping(raw_config.get("loss", {}), "loss", config_path)
    file_name = loss_config.get("name", "mse")

    if explicit_loss is not None:
        name = explicit_loss
        params: Mapping[str, object] = {} if explicit_loss_params is None else explicit_loss_params
        params_origin = "loss_params"
    elif explicit_loss_params is not None:
        name = file_name
        params = explicit_loss_params
        params_origin = "loss_params"
    else:
        name = file_name
        params = _require_mapping(loss_config.get("params", {}), "loss.params", config_path)
        params_origin = "[loss.params]"

    def error_factory(message: str, category: type[Exception], *, rejection: str = "") -> Exception:
        del category
        if rejection == "name":
            return TrainingConfigError(f"{config_path}: [loss] name: {message}")
        if rejection == "params":
            return TrainingConfigError(f"{config_path}: {params_origin}: {message}")
        if params_origin == "loss_params":
            return TrainingConfigError(f"{config_path}: loss_params: {message}")
        # Parameter-value messages begin with their catalog parameter name, so
        # this prefix produces a TOML path such as ``loss.params.delta``. That
        # formatting convention is supplied by resolve_loss_selection.
        return TrainingConfigError(f"{config_path}: loss.params.{message}")

    return resolve_loss_selection(name, params, error_factory=error_factory)


def _resolve_metric_names(
    raw_config: Mapping[str, Any],
    explicit_metrics: Optional[Sequence[str]],
    *,
    config_path: Path,
) -> tuple[str, ...]:
    if explicit_metrics is not None:
        if isinstance(explicit_metrics, str):
            raise TrainingConfigError(
                f"{config_path}: metrics override must be a sequence of strings."
            )
        names = tuple(explicit_metrics)
    else:
        metrics_config = raw_config.get("metrics", {})
        raw_names = _require_mapping(metrics_config, "metrics", config_path).get("names", ())
        if isinstance(raw_names, str):
            raise TrainingConfigError(f"{config_path}: metrics.names must be a list of strings.")
        names = tuple(raw_names)

    if any(not isinstance(name, str) for name in names):
        raise TrainingConfigError(f"{config_path}: metrics.names must contain only strings.")
    if len(set(names)) != len(names):
        raise TrainingConfigError(f"{config_path}: metrics.names must not contain duplicates.")

    unknown = sorted(set(names) - METRIC_REGISTRY)
    if unknown:
        valid = ", ".join(sorted(METRIC_REGISTRY))
        raise TrainingConfigError(
            f"{config_path}: unsupported metric name(s) {unknown!r}; expected one of ({valid})."
        )
    return names


def _resolve_metric_specs(metric_names: Sequence[str], *, output_dim: object) -> MetricsMap:
    resolved: MetricsMap = {}
    for name in metric_names:
        if name == "r2":
            resolved[name] = MetricRegistry.create("r2", num_outputs=output_dim or 1)
        else:
            resolved[name] = name
    return resolved


def _resolve_tracking_config(
    raw_config: Mapping[str, Any],
    *,
    metric_names: Sequence[str],
    config_path: Path,
) -> TrackingConfig:
    tracking_values = dict(
        _require_mapping(raw_config.get("tracking", {}), "tracking", config_path)
    )
    if "tags" in tracking_values:
        tags = tracking_values["tags"]
        if isinstance(tags, str) or not isinstance(tags, Sequence):
            raise TrainingConfigError(f"{config_path}: tracking.tags must be a list of strings.")
        if any(not isinstance(tag, str) for tag in tags):
            raise TrainingConfigError(f"{config_path}: tracking.tags must contain only strings.")
        tracking_values["tags"] = tuple(tags)

    if "metrics" in tracking_values:
        metrics = tracking_values["metrics"]
        if not isinstance(metrics, list):
            raise TrainingConfigError(f"{config_path}: tracking.metrics must be a TOML array.")
        if any(not isinstance(name, str) for name in metrics):
            raise TrainingConfigError(f"{config_path}: tracking.metrics must contain only strings.")
        tracking_values["metrics"] = tuple(metrics)

    if "enabled" in tracking_values and not isinstance(tracking_values["enabled"], bool):
        raise TrainingConfigError(f"{config_path}: tracking.enabled must be a boolean.")

    for key in sorted(TRACKING_KEYS - {"enabled", "tags", "metrics"}):
        if (
            key in tracking_values
            and tracking_values[key] is not None
            and not isinstance(tracking_values[key], str)
        ):
            raise TrainingConfigError(f"{config_path}: tracking.{key} must be a string.")

    try:
        tracking_config = TrackingConfig(**tracking_values)
        tracking_config.validate()
        if tracking_config.enabled:
            dynamic_metrics = tuple(
                f"{prefix}/{name}" for prefix in ("train", "val") for name in metric_names
            )
            validate_workflow_metrics(
                tracking_config.metrics,
                dynamic_metrics,
                workflow="configured trainer",
            )
    except TypeError as exc:
        raise TrainingConfigError(f"{config_path}: invalid tracking configuration: {exc}") from exc
    return tracking_config


def _raise_missing(key_path: str, config_path: Path) -> None:
    raise TrainingConfigError(f"{config_path}: missing required configuration key '{key_path}'.")


def _raise_type_error(key_path: str, expected: str, value: Any, config_path: Path) -> None:
    raise TrainingConfigError(
        f"{config_path}: {key_path} must be {expected}, got {type(value).__name__}."
    )
