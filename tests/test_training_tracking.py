"""Tests for optional experiment tracking during training."""

# ruff: noqa: E402

from pathlib import Path
import math
import warnings

import pytest

from tests.support import require_modules

torch = pytest.importorskip("torch")
require_modules("torch_geometric", "torch_scatter")

from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from phylognn.training import TrainingConfig, load_training_config
from phylognn.training.tracking import (
    FIXED_METRIC_CATALOG,
    TrackingConfig,
    TrackingError,
    TrackingRunInfo,
    WandbTracker,
    build_epoch_metrics,
    build_experiment_config,
    build_final_metrics,
    build_status_metrics,
    filter_quantitative_metrics,
    sanitize_config_metadata,
    validate_workflow_metrics,
)
from phylognn.training.trainer import Trainer


class FakeTracker:
    def __init__(
        self,
        *,
        start_error: Exception | None = None,
        finish_error: Exception | None = None,
    ) -> None:
        self.start_error = start_error
        self.finish_error = finish_error
        self.started = False
        self.start_payloads = []
        self.metric_calls = []
        self.finish_calls = []

    def start(self, config):
        if self.start_error is not None:
            raise self.start_error
        self.started = True
        self.start_payloads.append(dict(config))
        return TrackingRunInfo(run_id="run-1", run_url="https://wandb.test/run-1")

    def log_metrics(self, metrics, *, step):
        self.metric_calls.append((step, dict(metrics)))

    def finish(self, status):
        self.finish_calls.append(status)
        if self.finish_error is not None:
            raise self.finish_error

    @property
    def ordered_payloads(self):
        """Return metric payloads in the order they were logged."""
        return [payload for _, payload in self.metric_calls]

    @property
    def quantitative_payloads(self):
        """Return payloads containing at least one quantitative field."""
        return [
            payload
            for payload in self.ordered_payloads
            if any(not _is_operational_key(key) for key in payload)
        ]

    @property
    def operational_payloads(self):
        """Return payloads containing only operational fields."""
        return [
            payload
            for payload in self.ordered_payloads
            if payload and all(_is_operational_key(key) for key in payload)
        ]


def _is_operational_key(key):
    return key == "status/state" or key.startswith("stage/")


class FakeRun:
    def __init__(self) -> None:
        self.id = "fake-id"
        self.url = "https://wandb.test/fake-id"
        self.name = "fake-name"
        self.log_calls = []
        self.finish_calls = []
        self.artifact_calls = []

    def log(self, metrics, *, step):
        self.log_calls.append((step, dict(metrics)))

    def finish(self, *, exit_code=None):
        self.finish_calls.append(exit_code)

    def log_artifact(self, *args, **kwargs):  # pragma: no cover - must never be called.
        self.artifact_calls.append((args, kwargs))


class FakeWandbModule:
    def __init__(self, *, init_error: Exception | None = None) -> None:
        self.init_error = init_error
        self.init_calls = []
        self.run = FakeRun()

    def init(self, **kwargs):
        if self.init_error is not None:
            raise self.init_error
        self.init_calls.append(kwargs)
        return self.run


class TinyRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, data):
        return self.linear(data.x)


class ExplodingTrainer(Trainer):
    def train_epoch(self, train_loader):
        raise RuntimeError("boom")


class InterruptingTrainer(Trainer):
    def train_epoch(self, train_loader):
        raise KeyboardInterrupt()


class CountingTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_epoch_calls = 0

    def train_epoch(self, train_loader):
        self.train_epoch_calls += 1
        return super().train_epoch(train_loader)


def _loader():
    samples = [
        Data(
            x=torch.tensor([[1.0]]),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            y=torch.tensor([[2.0]]),
        ),
        Data(
            x=torch.tensor([[2.0]]),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            y=torch.tensor([[4.0]]),
        ),
    ]
    return DataLoader(samples, batch_size=1, shuffle=False)


def _trainer(tmp_path: Path, *, tracker=None, tracking_config=None, cls=Trainer, epochs=2):
    torch.manual_seed(5)
    return cls(
        model=TinyRegressor(),
        config=TrainingConfig(
            epochs=epochs,
            batch_size=1,
            learning_rate=0.01,
            scheduler=None,
            save_dir=str(tmp_path / "checkpoints"),
            verbose=False,
            save_best_only=True,
        ),
        metrics={"mae": "mae"},
        tracking_config=tracking_config,
        tracking_metadata={"model.type": "TinyRegressor", "data.config_file": "training.toml"},
        tracker=tracker,
    )


def test_wandb_tracker_lifecycle_and_no_artifact_upload(monkeypatch):
    fake_wandb = FakeWandbModule()
    monkeypatch.setattr("phylognn.training.tracking.import_module", lambda name: fake_wandb)
    tracker = WandbTracker(
        TrackingConfig(
            enabled=True,
            project="phylognn",
            run_name="run",
            group="group",
            job_type="train",
            tags=("tag",),
        )
    )

    info = tracker.start({"training.epochs": 1, "data.config_file": "/tmp/training.toml"})
    tracker.log_metrics({"train/loss": 1.5}, step=1)
    tracker.finish("completed")

    assert info.run_id == "fake-id"
    assert fake_wandb.init_calls[0]["project"] == "phylognn"
    assert fake_wandb.init_calls[0]["config"]["data.config_file"] == "training.toml"
    assert fake_wandb.run.log_calls == [(1, {"train/loss": 1.5})]
    assert fake_wandb.run.finish_calls == [0]
    assert fake_wandb.run.artifact_calls == []


def test_trainer_logs_epoch_final_and_completed_status(tmp_path: Path):
    tracker = FakeTracker()
    trainer = _trainer(
        tmp_path,
        tracker=tracker,
        tracking_config=TrackingConfig(enabled=True, project="phylognn"),
        epochs=2,
    )

    trainer.fit(train_loader=_loader(), val_loader=_loader())

    assert tracker.started
    assert tracker.start_payloads[0]["model.type"] == "TinyRegressor"
    epoch_calls = [call for call in tracker.metric_calls if "train/loss" in call[1]]
    assert [step for step, _ in epoch_calls] == [1, 2]
    assert all("val/loss" in payload for _, payload in epoch_calls)
    assert all("train/lr" in payload for _, payload in epoch_calls)
    assert all("train/epoch_time_sec" in payload for _, payload in epoch_calls)
    assert all(
        "lr" not in payload and "epoch_time_sec" not in payload for _, payload in epoch_calls
    )
    assert tracker.metric_calls[-2][1]["final/best_epoch"] >= 1.0
    assert tracker.metric_calls[-1] == (trainer.current_epoch, {"status/state": "completed"})
    assert tracker.finish_calls == ["completed"]


def test_metric_payload_builders_use_stable_names_and_skip_missing_validation():
    epoch_payload = build_epoch_metrics(
        train_metrics={"loss": 2.0, "mae": 1.0},
        val_metrics=None,
        lr=0.01,
        epoch_time_sec=0.5,
    )
    assert epoch_payload == {
        "train/loss": 2.0,
        "train/mae": 1.0,
        "train/lr": 0.01,
        "train/epoch_time_sec": 0.5,
    }
    assert build_final_metrics(best_val_loss=0.25, best_epoch=3) == {
        "final/best_val_loss": 0.25,
        "final/best_epoch": 3,
    }
    assert build_status_metrics("failed") == {"status/state": "failed"}


def test_tracking_builders_emit_finite_payloads_and_preserve_run_identity():
    """Shared builders retain scalar-only metric and identity contracts."""
    epoch_payload = build_epoch_metrics(
        train_metrics={"loss": 1.25},
        val_metrics={"mse": 0.5},
        lr=0.01,
        epoch_time_sec=0.125,
    )
    final_payload = build_final_metrics(best_val_loss=0.5, best_epoch=2)
    identity = TrackingRunInfo(run_id="run-1", run_name="leaf-fit", run_url="https://test/run-1")

    assert all(math.isfinite(value) for value in epoch_payload.values())
    assert all(math.isfinite(value) for value in final_payload.values())
    assert [build_status_metrics(status) for status in ("completed", "failed", "interrupted")] == [
        {"status/state": "completed"},
        {"status/state": "failed"},
        {"status/state": "interrupted"},
    ]
    assert identity == TrackingRunInfo(
        run_id="run-1", run_name="leaf-fit", run_url="https://test/run-1"
    )


def test_tracking_builders_omit_nonfinite_quantitative_values():
    """External payloads never contain NaN or infinity values."""
    assert build_epoch_metrics(
        train_metrics={"loss": 1.0, "nan_metric": float("nan")},
        val_metrics={"loss": float("inf"), "mae": 0.25},
        lr=float("inf"),
        epoch_time_sec=float("nan"),
    ) == {"train/loss": 1.0, "val/mae": 0.25}
    assert build_final_metrics(best_val_loss=float("nan"), best_epoch=None) == {}


def test_leaf_regression_epoch_metrics_are_fixed_finite_selectable_names():
    """Leaf-regression epoch metrics use the shared finite allowlist contract."""
    leaf_metric_names = (
        "train/score",
        "val/score",
        "train/pearson_r",
        "val/pearson_r",
        "train/mae",
        "val/mae",
    )
    train_metrics = {"loss": 1.0, "score": 0.8, "pearson_r": 0.9, "mae": 0.25}
    val_metrics = {"loss": 1.5, "score": 0.7, "pearson_r": 0.85, "mae": 0.5}

    payload = build_epoch_metrics(
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        lr=0.01,
        epoch_time_sec=0.125,
    )

    assert all(name in FIXED_METRIC_CATALOG for name in leaf_metric_names)
    assert (
        validate_workflow_metrics(leaf_metric_names, workflow="leaf regression")
        == leaf_metric_names
    )
    assert all(math.isfinite(payload[name]) for name in leaf_metric_names)
    assert filter_quantitative_metrics(payload, ("train/score", "val/mae")) == {
        "train/score": 0.8,
        "val/mae": 0.5,
    }


@pytest.mark.parametrize(
    "selection, expected_epoch_keys, expected_final_keys",
    [
        (
            None,
            {"train/loss", "train/mae", "train/lr", "train/epoch_time_sec", "val/loss", "val/mae"},
            {"final/best_val_loss", "final/best_epoch"},
        ),
        (("train/loss", "val/mae"), {"train/loss", "val/mae"}, set()),
        ((), set(), set()),
        (("cv/mae",), set(), set()),
    ],
)
def test_trainer_metric_selection_filters_quantitative_payloads(
    tmp_path: Path, selection, expected_epoch_keys, expected_final_keys
):
    """Selection affects only applicable quantitative fields in a standard run."""
    tracker = FakeTracker()
    trainer = _trainer(
        tmp_path,
        tracker=tracker,
        tracking_config=TrackingConfig(enabled=True, project="phylognn", metrics=selection),
        epochs=1,
    )

    history = trainer.fit(train_loader=_loader(), val_loader=_loader())

    epoch_payloads = [
        payload for _, payload in tracker.metric_calls if "status/state" not in payload
    ]
    final_payloads = [
        payload for _, payload in tracker.metric_calls if "final/best_epoch" in payload
    ]
    assert (set(epoch_payloads[0]) if epoch_payloads else set()) == expected_epoch_keys
    assert (set(final_payloads[0]) if final_payloads else set()) == expected_final_keys
    assert tracker.metric_calls[-1][1] == {"status/state": "completed"}
    assert history["train_loss"]


def test_trainer_dynamic_selection_validates_before_start_and_is_deterministic(tmp_path: Path):
    """Configured dynamic names are valid; unavailable and unsafe names never start a run."""
    valid_trackers = []
    for run_index in range(2):
        tracker = FakeTracker()
        trainer = _trainer(
            tmp_path / str(run_index),
            tracker=tracker,
            tracking_config=TrackingConfig(
                enabled=True, project="phylognn", metrics=("train/mae", "val/mae")
            ),
            epochs=1,
        )
        trainer.fit(train_loader=_loader(), val_loader=_loader())
        valid_trackers.append(tracker)
    assert [set(tracker.metric_calls[0][1]) for tracker in valid_trackers] == [
        {"train/mae", "val/mae"},
        {"train/mae", "val/mae"},
    ]

    for selection, pattern in [
        (("train/not_configured",), "unknown"),
        (("train/api-key",), "sensitive"),
    ]:
        tracker = FakeTracker()
        with pytest.raises(TrackingError, match=pattern):
            _trainer(
                tmp_path / pattern,
                tracker=tracker,
                tracking_config=TrackingConfig(enabled=True, project="phylognn", metrics=selection),
                epochs=1,
            )
        assert tracker.started is False


def test_toml_and_python_tracking_selections_filter_identical_payloads(tmp_path: Path):
    """A TOML allowlist has the same filtering behavior as the Python API."""
    config_path = tmp_path / "training.toml"
    config_path.write_text(
        """
[model]
type = "GATBiLSTMNet"

[model.params]
input_dim = 1
output_dim = 1
temporal_mode = "none"

[training]
epochs = 1

[metrics]
names = ["mae"]

[tracking]
enabled = true
project = "phylognn"
metrics = ["train/loss", "val/mae"]
""",
        encoding="utf-8",
    )
    toml_selection = load_training_config(config_path).tracking_config
    python_selection = TrackingConfig(
        enabled=True,
        project="phylognn",
        metrics=("train/loss", "val/mae"),
    )
    payloads = []
    for name, selection in (("toml", toml_selection), ("python", python_selection)):
        tracker = FakeTracker()
        _trainer(tmp_path / name, tracker=tracker, tracking_config=selection, epochs=1).fit(
            train_loader=_loader(), val_loader=_loader()
        )
        payloads.append([payload for _, payload in tracker.metric_calls])

    assert [set(payload) for payload in payloads[0]] == [set(payload) for payload in payloads[1]]


def test_ten_completed_run_configs_are_comparable():
    key_sets = []
    for idx in range(10):
        config = build_experiment_config(
            model_type="GATBiLSTMNet",
            model_params={"input_dim": 4 + idx, "output_dim": 1},
            training_values={"epochs": 1 + idx, "learning_rate": 0.001},
            loss_name="mse",
            metric_names=("mse", "rmse"),
            tracking_config=TrackingConfig(
                enabled=True,
                project="phylognn",
                group="comparison",
                dataset_id="dataset-v1",
            ),
            config_path=Path(f"/tmp/run-{idx}/training.toml"),
        )
        key_sets.append(set(config))
        assert config["data.config_file"] == "training.toml"
    assert len({frozenset(keys) for keys in key_sets}) == 1


def test_disabled_tracking_does_not_import_wandb(monkeypatch, tmp_path: Path):
    def fail_import(name):
        raise AssertionError("wandb should not be imported when tracking is disabled")

    monkeypatch.setattr("phylognn.training.tracking.import_module", fail_import)
    trainer = _trainer(tmp_path, tracking_config=TrackingConfig(enabled=False), epochs=1)

    history = trainer.fit(train_loader=_loader())

    assert history["train_loss"]


def test_missing_wandb_guidance_mentions_wandb_extra(monkeypatch):
    def fail_import(name):
        raise ImportError("No module named 'wandb'")

    monkeypatch.setattr("phylognn.training.tracking.import_module", fail_import)
    tracker = WandbTracker(TrackingConfig(enabled=True, project="phylognn"))

    with pytest.raises(TrackingError) as exc_info:
        tracker.start({})

    message = str(exc_info.value).lower()
    assert "wandb" in message
    assert "extra" in message


def test_enabled_tracking_initialization_fails_before_first_epoch(tmp_path: Path):
    tracker = FakeTracker(start_error=TrackingError("missing project"))
    trainer = _trainer(
        tmp_path,
        tracker=tracker,
        tracking_config=TrackingConfig(enabled=True, project="phylognn"),
        cls=CountingTrainer,
        epochs=1,
    )

    with pytest.raises(TrackingError, match="missing project"):
        trainer.fit(train_loader=_loader())

    assert trainer.train_epoch_calls == 0


def test_failure_and_keyboard_interrupt_finish_statuses(tmp_path: Path):
    failed_tracker = FakeTracker()
    failed = _trainer(
        tmp_path / "failed",
        tracker=failed_tracker,
        tracking_config=TrackingConfig(enabled=True, project="phylognn"),
        cls=ExplodingTrainer,
    )
    with pytest.raises(RuntimeError, match="boom"):
        failed.fit(train_loader=_loader())
    assert failed_tracker.finish_calls == ["failed"]
    assert failed_tracker.metric_calls[-1] == (0, {"status/state": "failed"})

    interrupted_tracker = FakeTracker()
    interrupted = _trainer(
        tmp_path / "interrupted",
        tracker=interrupted_tracker,
        tracking_config=TrackingConfig(enabled=True, project="phylognn"),
        cls=InterruptingTrainer,
    )
    with pytest.raises(KeyboardInterrupt):
        interrupted.fit(train_loader=_loader())
    assert interrupted_tracker.finish_calls == ["interrupted"]
    assert interrupted_tracker.metric_calls[-1] == (0, {"status/state": "interrupted"})


def test_training_failure_remains_primary_when_tracking_finish_fails(tmp_path: Path):
    tracker = FakeTracker(finish_error=TrackingError("cleanup failed"))
    trainer = _trainer(
        tmp_path,
        tracker=tracker,
        tracking_config=TrackingConfig(enabled=True, project="phylognn"),
        cls=ExplodingTrainer,
    )

    with pytest.warns(UserWarning, match="cleanup failed"):
        with pytest.raises(RuntimeError, match="boom"):
            trainer.fit(train_loader=_loader())

    assert tracker.finish_calls == ["failed"]


def test_training_failure_with_successful_tracking_cleanup_emits_no_cleanup_warning(
    tmp_path: Path,
):
    tracker = FakeTracker()
    trainer = _trainer(
        tmp_path,
        tracker=tracker,
        tracking_config=TrackingConfig(enabled=True, project="phylognn"),
        cls=ExplodingTrainer,
    )

    with warnings.catch_warnings(record=True) as warnings_record:
        warnings.simplefilter("always")
        with pytest.raises(RuntimeError, match="boom"):
            trainer.fit(train_loader=_loader())

    assert warnings_record == []
    assert tracker.finish_calls == ["failed"]


def test_secret_metadata_rejected_and_paths_are_sanitized():
    with pytest.raises(TrackingError, match="sensitive"):
        sanitize_config_metadata({"wandb.token": "secret"})

    sanitized = sanitize_config_metadata(
        {
            "data.config_file": "/Users/me/private/training.toml",
            "training.save_dir": "/Users/me/private/checkpoints",
            "data.dataset_id": "dataset-v1",
        }
    )
    assert sanitized == {
        "data.config_file": "training.toml",
        "data.dataset_id": "dataset-v1",
        "training.save_dir": "checkpoints",
    }
