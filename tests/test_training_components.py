"""Contract tests for dataset, trainer, and metric internals."""

from pathlib import Path

import pytest

from tests.support import require_modules


torch = pytest.importorskip("torch")
require_modules("torch_geometric")

from phylognn.training.dataset import DatasetSplit
from phylognn.training.metrics import mae_metric, mse_metric, relative_error_metric, rmse_metric
from phylognn.training.trainer import TrainingConfig, _detach_item, _safe_mean, _sanitize_task_name


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
    """Internal helpers should keep names safe and scalar conversions explicit."""
    assert _sanitize_task_name("task 1/alpha") == "task_1_alpha"
    assert _detach_item(torch.tensor(3.5)) == 3.5
    assert _safe_mean(9.0, 3) == 3.0

    with pytest.raises(ValueError, match="zero batches"):
        _safe_mean(1.0, 0)


def test_metric_helpers_return_scalar_tensors():
    """Public metrics should remain scalar-valued helpers."""
    pred = torch.tensor([[1.0], [2.0]])
    target = torch.tensor([[0.5], [2.5]])

    for metric in (mse_metric, mae_metric, rmse_metric, relative_error_metric):
        value = metric(pred, target)
        assert value.ndim == 0
