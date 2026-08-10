"""Contract tests for dataset, trainer, and metric internals."""

# ruff: noqa: E402

from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

from tests.support import require_modules

torch = pytest.importorskip("torch")
require_modules("torch_geometric")
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torchmetrics import MeanAbsolutePercentageError, MeanSquaredError, Metric, R2Score

from phylognn.training.dataset import DatasetSplit, SplitPhyloDataset, SplitPhyloDiskDataset
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


def _train_loader(num_samples: int = 1) -> DataLoader:
    return DataLoader(
        [
            Data(
                x=torch.ones((1, 1)),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                y=torch.ones(1),
            )
            for _ in range(num_samples)
        ],
        batch_size=1,
    )


def _counting_transform(name: str, calls: list[str]):
    def transform(data: Data) -> Data:
        calls.append(name)
        existing = list(getattr(data, "transform_log", []))
        data.transform_log = existing + [name]
        return data

    return transform


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


def test_split_phylo_dataset_applies_transform_exactly_once_and_preserves_plain_retrieval():
    graph = Data(x=torch.ones((1, 1)), edge_index=torch.empty((2, 0), dtype=torch.long))
    calls = []

    transformed = SplitPhyloDataset(
        data_list=[graph],
        labels=torch.tensor([[2.0]]),
        sample_ids=["sample-1"],
        transform=_counting_transform("base", calls),
    )[0]
    plain = SplitPhyloDataset(
        data_list=[graph],
        labels=torch.tensor([[2.0]]),
        sample_ids=["sample-1"],
    )[0]

    assert calls == ["base"]
    assert transformed.transform_log == ["base"]
    assert transformed.sample_id == "sample-1"
    assert torch.equal(transformed.y, torch.tensor([2.0]))
    assert not hasattr(plain, "transform_log")
    assert plain.sample_id == "sample-1"
    assert torch.equal(plain.x, graph.x)


def test_split_phylo_disk_dataset_applies_transform_exactly_once(tmp_path):
    graph_dir = tmp_path / "graphs"
    label_dir = tmp_path / "labels"
    graph_dir.mkdir()
    label_dir.mkdir()
    torch.save(
        Data(x=torch.ones((1, 1)), edge_index=torch.empty((2, 0), dtype=torch.long)),
        graph_dir / "sample.pt",
    )
    torch.save(torch.tensor([3.0]), label_dir / "sample.pt")
    calls = []

    dataset = SplitPhyloDiskDataset(
        graph_dir=graph_dir,
        label_dir=label_dir,
        transform=_counting_transform("disk", calls),
    )

    loaded = dataset[0]

    assert calls == ["disk"]
    assert loaded.transform_log == ["disk"]
    assert loaded.sample_id == "sample"
    assert torch.equal(loaded.y, torch.tensor([3.0]))


def test_split_dataset_view_applies_base_then_view_transform_once_each():
    graph = Data(x=torch.ones((1, 1)), edge_index=torch.empty((2, 0), dtype=torch.long))
    calls = []
    base_dataset = SplitPhyloDataset(
        data_list=[graph],
        sample_ids=["sample-1"],
        transform=_counting_transform("base", calls),
    )
    split = DatasetSplit.from_dict({"train": ["sample-1"]})
    view = base_dataset.subset(
        "train",
        split,
        transform=_counting_transform("view", calls),
    )

    loaded = view[0]

    assert calls == ["base", "view"]
    assert loaded.transform_log == ["base", "view"]
    assert loaded.sample_id == "sample-1"


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


def test_save_best_only_without_validation_saves_latest_checkpoint(tmp_path):
    trainer = Trainer(
        model=TinyRegressor(),
        config=TrainingConfig(
            epochs=1,
            batch_size=1,
            scheduler=None,
            save_dir=str(tmp_path / "checkpoints"),
            save_best_only=True,
            verbose=False,
        ),
    )

    with pytest.warns(UserWarning, match="checkpoint_latest.pt"):
        trainer.fit(train_loader=_train_loader())

    assert (trainer.save_dir / "checkpoint_latest.pt").is_file()
    assert not (trainer.save_dir / "best_model.pt").exists()
    assert not list(trainer.save_dir.glob("checkpoint_epoch_*.pt"))


def test_save_best_only_without_validation_overwrites_latest_checkpoint(tmp_path):
    trainer = Trainer(
        model=TinyRegressor(),
        config=TrainingConfig(
            epochs=2,
            batch_size=1,
            scheduler=None,
            save_dir=str(tmp_path / "checkpoints"),
            save_best_only=True,
            verbose=False,
        ),
    )

    with pytest.warns(UserWarning, match="checkpoint_latest.pt"):
        trainer.fit(train_loader=_train_loader())

    checkpoint_files = sorted(path.name for path in trainer.save_dir.glob("*.pt"))
    assert checkpoint_files == ["checkpoint_latest.pt", "final_model.pt"]
    checkpoint = torch.load(
        trainer.save_dir / "checkpoint_latest.pt",
        map_location=trainer.device,
        weights_only=False,
    )
    assert checkpoint["current_epoch"] == 2


# ---------------------------------------------------------------------------
# T003: Shared loss catalog — names, case sensitivity, rejection table
# ---------------------------------------------------------------------------

from phylognn.training.losses import (
    build_loss,
    format_loss_identifier,
    resolve_loss_selection,
    supported_loss_names,
)


class TestLossCatalogDiscovery:
    """T003: Public query and catalog immutability."""

    def test_supported_loss_names_returns_sorted_tuple(self):
        assert supported_loss_names() == ("huber", "mae", "mse")

    def test_supported_loss_names_returns_fresh_tuple(self):
        a = supported_loss_names()
        b = supported_loss_names()
        assert a is not b

    def test_case_sensitive_lookup_rejects_uppercase(self):
        with pytest.raises(ValueError, match="unsupported loss"):
            resolve_loss_selection("MSE", None)


class TestLossCatalogRejection:
    """T003: Full rejection table from contracts/loss_catalog.md."""

    @pytest.mark.parametrize("name", ["mse", "huber"])
    def test_non_string_parameter_key_is_reported_as_unknown(self, name):
        """Parameter keys must be named explicitly in shared-catalog errors."""
        with pytest.raises(ValueError, match=rf"unknown parameter.*1.*loss '{name}'"):
            resolve_loss_selection(name, {1: 2.0})

    def test_unsupported_name_lists_supported_identifiers(self):
        with pytest.raises(ValueError, match="huber.*mae.*mse"):
            resolve_loss_selection("logcosh", None)

    def test_non_string_name_raises_type_error(self):
        with pytest.raises(TypeError, match="must be a string"):
            resolve_loss_selection(42, None)

    def test_unknown_parameter_key_rejected(self):
        with pytest.raises(ValueError, match="unknown parameter.*beta"):
            resolve_loss_selection("huber", {"beta": 1.0})

    def test_parameter_supplied_for_parameter_free_loss(self):
        with pytest.raises(ValueError, match="does not accept parameters"):
            resolve_loss_selection("mse", {"delta": 1.0})

    def test_mae_rejects_delta(self):
        with pytest.raises(ValueError, match="does not accept parameters"):
            resolve_loss_selection("mae", {"delta": 1.0})

    def test_delta_boolean_rejected_as_wrong_type(self):
        with pytest.raises(TypeError, match="must be a real number"):
            resolve_loss_selection("huber", {"delta": True})

    def test_delta_string_rejected_as_wrong_type(self):
        with pytest.raises(TypeError, match="must be a real number"):
            resolve_loss_selection("huber", {"delta": "1.0"})

    def test_delta_zero_rejected(self):
        with pytest.raises(ValueError, match="strictly positive"):
            resolve_loss_selection("huber", {"delta": 0})

    def test_delta_negative_rejected(self):
        with pytest.raises(ValueError, match="strictly positive"):
            resolve_loss_selection("huber", {"delta": -1.0})

    def test_delta_nan_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            resolve_loss_selection("huber", {"delta": float("nan")})

    def test_delta_inf_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            resolve_loss_selection("huber", {"delta": float("inf")})


class TestLossCatalogResolution:
    """T003: Valid resolution paths."""

    def test_mse_with_none_params(self):
        name, params = resolve_loss_selection("mse", None)
        assert name == "mse"
        assert params == {}

    def test_mae_with_empty_params(self):
        name, params = resolve_loss_selection("mae", {})
        assert name == "mae"
        assert params == {}

    def test_huber_with_none_defaults_to_1(self):
        name, params = resolve_loss_selection("huber", None)
        assert name == "huber"
        assert params == {"delta": 1.0}

    def test_huber_with_explicit_delta(self):
        name, params = resolve_loss_selection("huber", {"delta": 2.0})
        assert name == "huber"
        assert params == {"delta": 2.0}

    def test_huber_integer_delta_normalized_to_float(self):
        _, params = resolve_loss_selection("huber", {"delta": 2})
        assert params["delta"] == 2.0
        assert isinstance(params["delta"], float)

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (np.float32(1.5), 1.5),
            (np.int64(2), 2.0),
            (Fraction(3, 2), 1.5),
        ],
    )
    def test_real_number_delta_types_normalize_on_both_surfaces(self, value, expected):
        """Every accepted real-number delta normalizes to a plain float."""
        from phylognn import LeafRegressionConfig

        _, params = resolve_loss_selection("huber", {"delta": value})
        config = LeafRegressionConfig(loss="huber", huber_delta=value)

        assert params == {"delta": expected}
        assert config.huber_delta == expected
        assert isinstance(params["delta"], float)
        assert isinstance(config.huber_delta, float)

    @pytest.mark.parametrize("value", [Decimal("1.5"), True])
    def test_non_real_delta_types_remain_rejected(self, value):
        """Decimal and bool do not satisfy the accepted real-number contract."""
        with pytest.raises(TypeError, match="must be a real number"):
            resolve_loss_selection("huber", {"delta": value})


class TestLossCatalogErrorFactory:
    """T003: Custom error_factory wiring."""

    def test_error_factory_receives_category_type_error(self):
        class CustomError(Exception):
            pass

        def factory(msg, category):
            assert category is TypeError
            return CustomError(msg)

        with pytest.raises(CustomError):
            resolve_loss_selection(42, None, error_factory=factory)

    def test_error_factory_receives_category_value_error(self):
        class CustomError(Exception):
            pass

        def factory(msg, category):
            assert category is ValueError
            return CustomError(msg)

        with pytest.raises(CustomError):
            resolve_loss_selection("logcosh", None, error_factory=factory)

    def test_two_argument_factory_type_error_propagates_after_one_call(self):
        """A callback TypeError is not mistaken for a signature mismatch."""
        calls = 0

        def factory(msg, category):
            nonlocal calls
            calls += 1
            raise TypeError("factory body failure")

        with pytest.raises(TypeError, match="factory body failure"):
            resolve_loss_selection("logcosh", None, error_factory=factory)

        assert calls == 1

    @pytest.mark.parametrize(
        ("name", "params", "expected_rejection"),
        [
            ("logcosh", None, "name"),
            ("mae", {"delta": 1.0}, "params"),
            ("huber", {"delta": 0}, "param_value"),
        ],
    )
    def test_error_factory_receives_structured_rejection(self, name, params, expected_rejection):
        """Factories accepting the optional keyword receive the rejection class."""
        seen = []

        class CustomError(Exception):
            pass

        def factory(msg, category, *, rejection):
            seen.append(rejection)
            return CustomError(msg)

        with pytest.raises(CustomError):
            resolve_loss_selection(name, params, error_factory=factory)

        assert seen == [expected_rejection]


# ---------------------------------------------------------------------------
# T009: Numeric pinning, mse equivalence, requires_grad, identifier text
# ---------------------------------------------------------------------------


class TestBuildLossNumerics:
    """T009: Pinned numeric values and loss construction contracts."""

    def test_huber_pinned_value(self):
        """Predictions [0, 0], targets [1, 5], delta=2.0 -> 4.25."""
        loss_fn = build_loss("huber", {"delta": 2.0})
        predictions = torch.tensor([0.0, 0.0])
        targets = torch.tensor([1.0, 5.0])
        result = loss_fn(predictions, targets)
        assert result.item() == pytest.approx(4.25)

    def test_huber_mean_reduction_invariant_to_repetition(self):
        """Repeating the residual pattern must not change the value (mean reduction)."""
        loss_fn = build_loss("huber", {"delta": 2.0})
        predictions_2 = torch.tensor([0.0, 0.0])
        targets_2 = torch.tensor([1.0, 5.0])
        predictions_4 = torch.tensor([0.0, 0.0, 0.0, 0.0])
        targets_4 = torch.tensor([1.0, 5.0, 1.0, 5.0])
        val_2 = loss_fn(predictions_2, targets_2).item()
        val_4 = loss_fn(predictions_4, targets_4).item()
        assert val_2 == pytest.approx(4.25)
        assert val_4 == pytest.approx(4.25)

    def test_mse_matches_functional(self):
        """Built mse must be numerically identical to torch.nn.functional.mse_loss."""
        loss_fn = build_loss("mse", {})
        predictions = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.5, 2.5, 2.0])
        expected = torch.nn.functional.mse_loss(predictions, targets)
        assert loss_fn(predictions, targets).item() == pytest.approx(expected.item())

    def test_requires_grad_propagates(self):
        """The loss tensor must have requires_grad=True when predictions do."""
        loss_fn = build_loss("huber", {"delta": 1.0})
        predictions = torch.tensor([1.0, 2.0], requires_grad=True)
        targets = torch.tensor([0.5, 2.5])
        result = loss_fn(predictions, targets)
        assert result.requires_grad is True


class TestFormatLossIdentifier:
    """T009: Identifier rendering tests."""

    def test_mse_bare_name(self):
        assert format_loss_identifier("mse", {}) == "mse"

    def test_mae_bare_name(self):
        assert format_loss_identifier("mae", {}) == "mae"

    def test_huber_default_delta_from_int(self):
        """Integer 1 and float 1.0 both render as huber(delta=1.0)."""
        assert format_loss_identifier("huber", {"delta": 1}) == "huber(delta=1.0)"
        assert format_loss_identifier("huber", {"delta": 1.0}) == "huber(delta=1.0)"

    def test_huber_custom_delta(self):
        assert format_loss_identifier("huber", {"delta": 1.5}) == "huber(delta=1.5)"

    def test_huber_small_delta(self):
        assert format_loss_identifier("huber", {"delta": 0.001}) == "huber(delta=0.001)"
