"""Contract tests for dataset, trainer, and metric internals."""

# ruff: noqa: E402

import pytest

from tests.support import require_modules

torch = pytest.importorskip("torch")
require_modules("torch_geometric")
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torchmetrics import MeanAbsolutePercentageError, MeanSquaredError, Metric, R2Score

from phylognn.training.dataset import DatasetSplit, SplitPhyloDiskDataset
from phylognn.training.metrics import MetricRegistry
from phylognn.training.trainer import Trainer, TrainingConfig, _detach_item, _safe_mean


class TinyRegressor(nn.Module):
    def __init__(self, output_dim: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(1, output_dim)

    def forward(self, data: Data) -> torch.Tensor:
        return self.linear(data.x)


class ConstantOutputModel(nn.Module):
    def __init__(self, output: torch.Tensor) -> None:
        super().__init__()
        self.output = nn.Parameter(output.clone())

    def forward(self, data: Data) -> torch.Tensor:
        return self.output


def _trainer(tmp_path, *, metrics=None, output_dim: int = 1) -> Trainer:
    return Trainer(
        model=TinyRegressor(output_dim=output_dim),
        config=TrainingConfig(
            epochs=1,
            batch_size=2,
            scheduler=None,
            save_dir=str(tmp_path / "checkpoints"),
            verbose=False,
        ),
        metrics={} if metrics is None else metrics,
    )


def _constant_trainer(tmp_path, output: torch.Tensor, *, metrics=None) -> Trainer:
    return Trainer(
        model=ConstantOutputModel(output),
        config=TrainingConfig(
            epochs=1,
            batch_size=2,
            scheduler=None,
            save_dir=str(tmp_path / "checkpoints"),
            verbose=False,
        ),
        metrics={} if metrics is None else metrics,
    )


def _loader_with_target(target: torch.Tensor) -> DataLoader:
    return DataLoader(
        [Data(x=torch.ones((1, 1)), edge_index=torch.empty((2, 0), dtype=torch.long), y=target)]
    )


def test_dataset_split_from_dict_preserves_names_and_membership():
    """Explicit split construction should preserve insertion order."""
    split = DatasetSplit.from_dict({"train": ["a", "b"], "val": ["c"]})

    assert split.split_names() == ["train", "val"]
    assert split.sample_ids("train") == ["a", "b"]
    assert split.contains("c")


def test_dataset_split_rejects_duplicate_sample_ids():
    """A sample cannot belong to multiple splits."""
    with pytest.raises(ValueError, match="appears in multiple splits"):
        DatasetSplit.from_dict({"train": ["a"], "test": ["a"]})


def test_dataset_split_from_ratios_is_deterministic():
    """Ratio-based splitting should be deterministic for a fixed seed."""
    left = DatasetSplit.from_ratios(
        sample_ids=["a", "b", "c", "d"],
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        seed=7,
    )
    right = DatasetSplit.from_ratios(
        sample_ids=["a", "b", "c", "d"],
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        seed=7,
    )

    assert left.splits == right.splits


def test_training_config_rejects_unknown_optimizer_and_scheduler():
    """Configuration should fail fast on unsupported optimizer choices."""
    with pytest.raises(ValueError, match="optimizer"):
        TrainingConfig(optimizer="bogus").validate()

    with pytest.raises(ValueError, match="scheduler"):
        TrainingConfig(scheduler="bogus").validate()


def test_trainer_helper_functions_preserve_scalar_contracts():
    """Internal helpers should keep scalar conversions explicit."""
    assert _detach_item(torch.tensor(3.5)) == 3.5
    assert _safe_mean(9.0, 3) == 3.0

    with pytest.raises(ValueError, match="zero batches"):
        _safe_mean(1.0, 0)


@pytest.mark.parametrize(
    ("metric_name", "expected"),
    [
        ("mse", 0.25),
        ("mae", 0.5),
        ("rmse", 0.5),
        ("r2", 0.75),
    ],
)
def test_trainer_computes_builtin_torchmetrics(tmp_path, metric_name: str, expected: float):
    """Trainer-managed metrics should aggregate through TorchMetrics state."""
    pred = torch.tensor([[1.0], [2.0]])
    target = torch.tensor([[0.5], [2.5]])
    trainer = _trainer(tmp_path, metrics={metric_name: metric_name})

    trainer._update_metrics(pred, target)
    values = trainer._compute_metrics()

    assert values[metric_name] == pytest.approx(expected)


def test_configured_multi_output_r2_matches_torchmetrics_raw_value_average(tmp_path):
    """Configured multi-output R2 should validate dimensions and use TorchMetrics semantics."""
    pred = torch.tensor([[1.0, 2.0], [3.0, 5.0], [6.0, 7.0]])
    target = torch.tensor([[1.0, 1.0], [2.0, 5.0], [7.0, 9.0]])
    metric = MetricRegistry.create("r2", num_outputs=2)
    expected_metric = R2Score(multioutput="raw_values")
    expected_metric.update(pred, target)
    expected = expected_metric.compute().mean().item()
    trainer = _trainer(tmp_path, metrics={"r2": metric}, output_dim=2)

    trainer._update_metrics(pred, target)
    values = trainer._compute_metrics()

    assert values["r2"] == pytest.approx(expected)


def test_mape_zero_targets_match_torchmetrics_behavior(tmp_path):
    """MAPE should use TorchMetrics zero-target semantics rather than the legacy formula."""
    pred = torch.tensor([[0.0], [1.0]])
    target = torch.tensor([[0.0], [0.0]])
    expected_metric = MeanAbsolutePercentageError()
    expected_metric.update(pred, target)
    trainer = _trainer(tmp_path, metrics={"mape": "mape"})

    trainer._update_metrics(pred, target)
    values = trainer._compute_metrics()

    assert torch.isfinite(torch.tensor(values["mape"]))
    assert values["mape"] == pytest.approx(expected_metric.compute().item())


def test_registry_rejects_invalid_r2_num_outputs():
    with pytest.raises(ValueError, match="positive integer"):
        MetricRegistry.create("r2", num_outputs=0)


def test_trainer_rejects_non_torchmetrics_custom_metric(tmp_path):
    with pytest.raises(TypeError, match="torchmetrics.Metric"):
        _trainer(tmp_path, metrics={"bad": lambda pred, target: torch.tensor(0.0)})


def test_trainer_rejects_r2_output_dimension_mismatch(tmp_path):
    trainer = _trainer(tmp_path, metrics={"r2": MetricRegistry.create("r2", num_outputs=2)})

    with pytest.raises(ValueError, match="configured for 2 output"):
        trainer._update_metrics(torch.tensor([[1.0], [2.0]]), torch.tensor([[1.0], [2.0]]))


def test_trainer_rejects_distributed_metrics_with_step_sync(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    with pytest.raises(ValueError, match="dist_sync_on_step=False"):
        _trainer(tmp_path, metrics={"mse": MeanSquaredError(dist_sync_on_step=True)})


def test_registry_created_metrics_disable_step_sync_for_ddp():
    """DDP sync is deferred to compute time by disabling per-step synchronization."""
    for name in MetricRegistry.names():
        metric = MetricRegistry.create(name)
        assert isinstance(metric, Metric)
        assert metric.dist_sync_on_step is False


def test_validate_single_output_accepts_vector_targets(tmp_path):
    """Single-output predictions should not broadcast against [batch] targets."""
    output = torch.tensor([[1.0], [2.0]])
    trainer = _constant_trainer(tmp_path, output, metrics={"mse": "mse"})

    result = trainer.validate(_loader_with_target(torch.tensor([1.0, 2.0])))

    assert result["loss"] == pytest.approx(0.0)
    assert result["mse"] == pytest.approx(0.0)


def test_validate_single_output_accepts_column_targets(tmp_path):
    """Single-output predictions should accept already aligned [batch, 1] targets."""
    output = torch.tensor([[1.0], [2.0]])
    trainer = _constant_trainer(tmp_path, output, metrics={"mse": "mse"})

    result = trainer.validate(_loader_with_target(torch.tensor([[1.0], [2.0]])))

    assert result["loss"] == pytest.approx(0.0)
    assert result["mse"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("prediction", "target"),
    [
        (torch.ones((2, 1)), torch.ones((1, 2))),
        (torch.ones((2, 2)), torch.ones(2)),
        (torch.ones((2, 2)), torch.ones((2, 1))),
    ],
)
def test_trainer_rejects_incompatible_target_shapes(tmp_path, prediction, target):
    """Ambiguous target shapes should fail before loss broadcasting can occur."""
    trainer = _constant_trainer(tmp_path, prediction)

    with pytest.raises(ValueError, match="Prediction and target"):
        trainer.validate(_loader_with_target(target))


def test_trainer_does_not_update_metrics_after_target_shape_error(tmp_path):
    """Shape validation must happen before metric state is mutated."""
    trainer = _constant_trainer(tmp_path, torch.ones((2, 2)), metrics={"mse": "mse"})

    with pytest.raises(ValueError, match="Prediction and target shapes are incompatible"):
        trainer.validate(_loader_with_target(torch.ones((2, 1))))

    metric = trainer.metrics["mse"]
    assert metric.update_count == 0


def test_split_phylo_disk_dataset_loaders_use_explicit_trusted_load(tmp_path, monkeypatch):
    """Disk graph and label artifacts should opt into complete-object loading explicitly."""
    graph_dir = tmp_path / "graphs"
    label_dir = tmp_path / "labels"
    graph_dir.mkdir()
    label_dir.mkdir()
    graph_path = graph_dir / "sample.pt"
    label_path = label_dir / "sample.pt"
    graph_path.touch()
    label_path.touch()
    graph = Data(x=torch.ones((1, 1)), edge_index=torch.empty((2, 0), dtype=torch.long))
    label = torch.tensor([3.0])
    calls = []

    def fake_load(path, *, map_location=None, weights_only=None):
        calls.append((path, map_location, weights_only))
        return graph if path == graph_path else label

    monkeypatch.setattr(torch, "load", fake_load)

    dataset = SplitPhyloDiskDataset(graph_dir=graph_dir, label_dir=label_dir)
    loaded = dataset[0]

    assert torch.equal(loaded.y, label)
    assert calls == [
        (graph_path, "cpu", False),
        (label_path, "cpu", False),
    ]


def test_trainer_load_checkpoint_uses_explicit_trusted_load(tmp_path, monkeypatch):
    """Trainer checkpoints should opt into complete-object loading explicitly."""
    trainer = _trainer(tmp_path)
    checkpoint = trainer._checkpoint_state()
    checkpoint["current_epoch"] = 1
    calls = []

    def fake_load(path, *, map_location=None, weights_only=None):
        calls.append((path, map_location, weights_only))
        return checkpoint

    monkeypatch.setattr(torch, "load", fake_load)

    trainer.load_checkpoint("model.pt")

    assert trainer.current_epoch == 1
    assert calls == [(trainer.save_dir / "model.pt", trainer.device, False)]
