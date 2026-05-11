"""Tests for TOML-backed training configuration setup."""

# ruff: noqa: E402

from pathlib import Path

import pytest

from tests.support import require_modules

pytest.importorskip("tomllib")
torch = pytest.importorskip("torch")
require_modules("torch_geometric", "torch_scatter")

from phylognn.models import GATBiLSTMNet
from phylognn.training import (
    ConfiguredTrainingSetup,
    Trainer,
    TrainingConfig,
    TrainingConfigError,
    TrackingConfig,
    create_trainer_from_config,
    load_training_config,
)


def _write_config(tmp_path: Path, text: str, name: str = "training.toml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _minimal_config(*, input_dim: int = 8, output_dim: int = 1, epochs: int = 5) -> str:
    return f"""
[model]
type = "GATBiLSTMNet"

[model.params]
input_dim = {input_dim}
output_dim = {output_dim}
temporal_mode = "none"

[training]
epochs = {epochs}
batch_size = 16
learning_rate = 0.001

[loss]
name = "mse"

[metrics]
names = ["mse", "mae"]
"""


def test_load_training_config_builds_minimal_setup(tmp_path: Path):
    config_path = _write_config(tmp_path, _minimal_config())

    setup = load_training_config(config_path)

    assert isinstance(setup, ConfiguredTrainingSetup)
    assert isinstance(setup.model, GATBiLSTMNet)
    assert isinstance(setup.training_config, TrainingConfig)
    assert setup.model.input_dim == 8
    assert setup.model.output_dim == 1
    assert setup.model.temporal_mode == "none"
    assert setup.training_config.epochs == 5
    assert setup.training_config.batch_size == 16
    assert type(setup.loss_fn).__name__ == "MSELoss"
    assert set(setup.metrics) == {"mse", "mae"}
    assert setup.tracking_config == TrackingConfig()
    assert setup.tracking_metadata["data.config_file"] == "training.toml"


def test_load_training_config_applies_defaults_for_omitted_optional_values(tmp_path: Path):
    config_path = _write_config(
        tmp_path,
        """
[model]
type = "GATBiLSTMNet"

[model.params]
input_dim = 4
output_dim = 2
temporal_mode = "none"

[training]
epochs = 1
""",
    )

    setup = load_training_config(config_path)

    assert setup.training_config.batch_size == TrainingConfig().batch_size
    assert setup.training_config.optimizer == TrainingConfig().optimizer
    assert setup.training_config.scheduler == TrainingConfig().scheduler
    assert setup.training_config.early_stopping_patience is None
    assert type(setup.loss_fn).__name__ == "MSELoss"
    assert setup.metrics == {}
    assert setup.tracking_config.enabled is False


@pytest.mark.parametrize("value", ['"none"', "0", "-1"])
def test_load_training_config_maps_disabled_early_stopping_to_none(
    tmp_path: Path,
    value: str,
):
    config_path = _write_config(
        tmp_path,
        _minimal_config().replace(
            "learning_rate = 0.001",
            f"learning_rate = 0.001\nearly_stopping_patience = {value}",
        ),
    )

    setup = load_training_config(config_path)

    assert setup.training_config.early_stopping_patience is None


def test_load_training_config_preserves_positive_early_stopping_patience(tmp_path: Path):
    config_path = _write_config(
        tmp_path,
        _minimal_config().replace(
            "learning_rate = 0.001",
            "learning_rate = 0.001\nearly_stopping_patience = 3",
        ),
    )

    setup = load_training_config(config_path)

    assert setup.training_config.early_stopping_patience == 3


def test_load_training_config_configures_r2_for_output_dim(tmp_path: Path):
    config_path = _write_config(
        tmp_path,
        _minimal_config(output_dim=2).replace('names = ["mse", "mae"]', 'names = ["r2"]'),
    )

    setup = load_training_config(config_path)

    assert setup.metrics["r2"].num_outputs == 2


def test_load_training_config_configures_r2_with_default_num_outputs(tmp_path: Path):
    config_path = _write_config(
        tmp_path,
        _minimal_config().replace('names = ["mse", "mae"]', 'names = ["r2"]'),
    )

    setup = load_training_config(config_path)

    assert setup.metrics["r2"].num_outputs == 1


def test_create_trainer_from_config_leaves_data_to_caller(tmp_path: Path):
    config_path = _write_config(tmp_path, _minimal_config(epochs=1))

    trainer = create_trainer_from_config(
        config_path,
        training_overrides={"save_dir": str(tmp_path / "checkpoints"), "verbose": False},
    )

    assert isinstance(trainer, Trainer)
    assert trainer.config.epochs == 1
    assert trainer.tracking_config.enabled is False
    with pytest.raises(ValueError, match="train_dataset or train_loader"):
        trainer.fit()


def test_missing_file_and_malformed_toml_raise_training_config_error(tmp_path: Path):
    with pytest.raises(TrainingConfigError, match="cannot read TOML"):
        load_training_config(tmp_path / "missing.toml")

    malformed_path = _write_config(tmp_path, "[model\n", name="bad.toml")
    with pytest.raises(TrainingConfigError, match="invalid TOML"):
        load_training_config(malformed_path)


@pytest.mark.parametrize(
    "text, pattern",
    [
        (
            """
[training]
epochs = 1
""",
            "model",
        ),
        (
            """
[model]
type = "GATBiLSTMNet"

[training]
epochs = 1
""",
            "model.params",
        ),
        (
            """
[model]
type = "GATBiLSTMNet"

[model.params]
output_dim = 1
temporal_mode = "none"

[training]
epochs = 1
""",
            "model.params.input_dim",
        ),
        (
            """
[model]
type = "GATBiLSTMNet"

[model.params]
input_dim = 4
temporal_mode = "none"

[training]
epochs = 1
""",
            "model.params.output_dim",
        ),
    ],
)
def test_missing_required_sections_and_model_params_fail(tmp_path: Path, text: str, pattern: str):
    config_path = _write_config(tmp_path, text)

    with pytest.raises(TrainingConfigError, match=pattern):
        load_training_config(config_path)


@pytest.mark.parametrize(
    "text, pattern",
    [
        (_minimal_config() + "\n[dataset]\npath = 'data'\n", "unknown key.*<root>"),
        (
            """
[model]
type = "GATBiLSTMNet"
unexpected = true

[model.params]
input_dim = 4
output_dim = 1
temporal_mode = "none"

[training]
epochs = 1
""",
            "unknown key.*model",
        ),
        (
            """
[model]
type = "GATBiLSTMNet"

[model.params]
input_dim = 4
output_dim = 1
temporal_mode = "none"
unknown_param = 1

[training]
epochs = 1
""",
            "unknown key.*model.params",
        ),
        (
            _minimal_config().replace(
                "learning_rate = 0.001", "learning_rate = 0.001\nunknown = 1"
            ),
            "unknown key.*training",
        ),
        (
            _minimal_config().replace('name = "mse"', 'name = "mse"\nextra = true'),
            "unknown key.*loss",
        ),
        (
            _minimal_config().replace('names = ["mse", "mae"]', 'names = ["mse"]\nextra = true'),
            "unknown key.*metrics",
        ),
    ],
)
def test_unknown_keys_fail_before_training(tmp_path: Path, text: str, pattern: str):
    config_path = _write_config(tmp_path, text)

    with pytest.raises(TrainingConfigError, match=pattern):
        load_training_config(config_path)


@pytest.mark.parametrize(
    "text, pattern",
    [
        (_minimal_config().replace('"GATBiLSTMNet"', '"OtherModel"'), "model.type"),
        (_minimal_config().replace('name = "mse"', 'name = "huber"'), "loss.name"),
        (_minimal_config().replace('"mae"', '"bad_metric"'), "unsupported metric"),
        (_minimal_config().replace('["mse", "mae"]', '["mse", "mse"]'), "duplicates"),
    ],
)
def test_unsupported_model_loss_metric_and_duplicate_metrics_fail(
    tmp_path: Path,
    text: str,
    pattern: str,
):
    config_path = _write_config(tmp_path, text)

    with pytest.raises(TrainingConfigError, match=pattern):
        load_training_config(config_path)


@pytest.mark.parametrize(
    "text, pattern",
    [
        (_minimal_config(input_dim=0), "input_dim"),
        (_minimal_config(epochs=0), "epochs"),
        (
            _minimal_config().replace("learning_rate = 0.001", 'learning_rate = "fast"'),
            "learning_rate",
        ),
        (
            _minimal_config().replace('temporal_mode = "none"', 'temporal_mode = "lstm"'),
            "num_time_bins",
        ),
        (
            _minimal_config().replace(
                'temporal_mode = "none"',
                'temporal_mode = "none"\ntemporal_fc_hidden_dims = [8, 0]',
            ),
            "temporal_fc_hidden_dims",
        ),
    ],
)
def test_invalid_types_and_numeric_ranges_fail(tmp_path: Path, text: str, pattern: str):
    config_path = _write_config(tmp_path, text)

    with pytest.raises(TrainingConfigError, match=pattern):
        load_training_config(config_path)


def test_two_config_files_return_distinct_effective_setups(tmp_path: Path):
    first_path = _write_config(tmp_path, _minimal_config(input_dim=4, epochs=1), name="first.toml")
    second_path = _write_config(
        tmp_path, _minimal_config(input_dim=6, epochs=2), name="second.toml"
    )

    first = load_training_config(first_path)
    second = load_training_config(second_path)

    assert first is not second
    assert first.model is not second.model
    assert first.training_config is not second.training_config
    assert first.metrics is not second.metrics
    assert first.model.input_dim == 4
    assert second.model.input_dim == 6
    assert first.training_config.epochs == 1
    assert second.training_config.epochs == 2


def test_explicit_overrides_win_over_toml_values(tmp_path: Path):
    config_path = _write_config(tmp_path, _minimal_config(input_dim=4, epochs=5))

    setup = load_training_config(
        config_path,
        model_overrides={"input_dim": 7},
        training_overrides={"epochs": 2, "save_dir": str(tmp_path / "run")},
        loss="mae",
        metrics=["rmse", "mape"],
    )

    assert setup.model.input_dim == 7
    assert setup.training_config.epochs == 2
    assert setup.training_config.save_dir == str(tmp_path / "run")
    assert type(setup.loss_fn).__name__ == "L1Loss"
    assert list(setup.metrics) == ["rmse", "mape"]
    assert setup.metrics == {"rmse": "rmse", "mape": "mape"}


def test_legacy_metric_helper_names_are_not_registered(tmp_path: Path):
    config_path = _write_config(
        tmp_path,
        _minimal_config().replace('names = ["mse", "mae"]', 'names = ["relative_error"]'),
    )

    with pytest.raises(TrainingConfigError, match="unsupported metric"):
        load_training_config(config_path)


def test_tracking_section_builds_enabled_tracking_setup(tmp_path: Path):
    config_path = _write_config(
        tmp_path,
        _minimal_config(epochs=1) + """
[tracking]
enabled = true
backend = "wandb"
project = "phylognn"
run_name = "baseline"
group = "ablation"
job_type = "train"
tags = ["baseline", "gat"]
dataset_id = "dataset-v1"
""",
    )

    setup = load_training_config(config_path)
    trainer = create_trainer_from_config(
        config_path,
        training_overrides={"save_dir": str(tmp_path / "checkpoints"), "verbose": False},
    )

    assert setup.tracking_config.enabled is True
    assert setup.tracking_config.project == "phylognn"
    assert setup.tracking_config.tags == ("baseline", "gat")
    assert setup.tracking_metadata["tracking.group"] == "ablation"
    assert setup.tracking_metadata["tracking.tags"] == ("baseline", "gat")
    assert setup.tracking_metadata["data.dataset_id"] == "dataset-v1"
    assert setup.tracking_metadata["data.config_file"] == "training.toml"
    assert setup.tracking_metadata["training.save_dir"] == "checkpoints"
    assert trainer.tracking_config.enabled is True


@pytest.mark.parametrize(
    "tracking_text, pattern",
    [
        ("enabled = true\nproject = 123\n", "tracking.project"),
        ("enabled = true\nproject = 'phylognn'\ntags = 'bad'\n", "tracking.tags"),
        ("enabled = 'yes'\nproject = 'phylognn'\n", "tracking.enabled"),
        ("enabled = true\nproject = 'phylognn'\nunknown = true\n", "unknown key.*tracking"),
        ("enabled = true\n", "tracking.project"),
    ],
)
def test_invalid_tracking_config_fails_before_training(
    tmp_path: Path,
    tracking_text: str,
    pattern: str,
):
    config_path = _write_config(
        tmp_path,
        _minimal_config(epochs=1) + f"""
[tracking]
{tracking_text}
""",
    )

    with pytest.raises(TrainingConfigError, match=pattern):
        load_training_config(config_path)
