"""Tests for the recommended leaf-regression workflow."""

import random

import numpy as np
import pytest
import torch


class _ConfiguredRegressor(torch.nn.Module):
    instances: list["_ConfiguredRegressor"] = []

    def __init__(self, offset: float) -> None:
        super().__init__()
        self.offset = offset
        self.weight = torch.nn.Parameter(torch.tensor(0.1))
        type(self).instances.append(self)

    def forward(
        self, representations: torch.Tensor, position_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        predictions = representations[:, 0, 0] * self.weight + self.offset
        attention = position_mask.to(dtype=representations.dtype)
        attention = attention / attention.sum(dim=1, keepdim=True)
        return predictions, attention


class _PredictionOnlyRegressor(torch.nn.Module):
    """Small custom model that intentionally omits interpretation output."""

    def __init__(self, offset: float) -> None:
        super().__init__()
        self.offset = offset
        self.weight = torch.nn.Parameter(torch.tensor(0.1))

    def forward(self, representations: torch.Tensor, position_mask: torch.Tensor) -> torch.Tensor:
        del position_mask
        return representations[:, 0, 0] * self.weight + self.offset


class _NoParameterRegressor(torch.nn.Module):
    """Return valid predictions without offering any trainable parameter."""

    def forward(self, representations: torch.Tensor, position_mask: torch.Tensor) -> torch.Tensor:
        del position_mask
        return representations[:, 0, 0]


class _InvalidOutputRegressor(torch.nn.Module):
    """Produce configurable invalid outputs while retaining a trainable probe."""

    instances: list["_InvalidOutputRegressor"] = []

    def __init__(self, kind: str) -> None:
        super().__init__()
        self.kind = kind
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        type(self).instances.append(self)

    def forward(self, representations: torch.Tensor, position_mask: torch.Tensor) -> object:
        predictions = representations[:, 0, 0] * self.weight
        if self.kind == "shape":
            return predictions[:, None]
        if self.kind == "nonfinite":
            return predictions * float("nan")
        if self.kind == "nondifferentiable":
            return predictions.detach()
        if self.kind == "attention":
            return predictions, position_mask[:, :-1].to(dtype=predictions.dtype)
        return (predictions,)


class _ConstructorFailureRegressor(torch.nn.Module):
    """Represent a model class that cannot satisfy its construction contract."""

    def __init__(self) -> None:
        raise RuntimeError("construction failed")


class _ObservableLeafTracker:
    """In-memory tracker that exposes lifecycle calls for coordinator tests."""

    def __init__(
        self,
        *,
        fail_metric_key: str | None = None,
        fail_finish: bool = False,
    ) -> None:
        self.start_payloads: list[dict[str, object]] = []
        self.metric_calls: list[tuple[int, dict[str, float | int | str]]] = []
        self.finish_calls: list[str] = []
        self.fail_metric_key = fail_metric_key
        self.fail_finish = fail_finish

    def start(self, config):
        from phylognn.training.tracking import TrackingRunInfo

        self.start_payloads.append(dict(config))
        return TrackingRunInfo(run_id="leaf-run", run_name="leaf-name", run_url="https://test/run")

    def log_metrics(self, metrics, *, step):
        if self.fail_metric_key is not None and self.fail_metric_key in metrics:
            from phylognn.training.tracking import TrackingError

            raise TrackingError(f"metric failure for {self.fail_metric_key}")
        self.metric_calls.append((step, dict(metrics)))

    def finish(self, status):
        self.finish_calls.append(status)
        if self.fail_finish:
            from phylognn.training.tracking import TrackingError

            raise TrackingError("finish failure")


class _InterruptingRegressor(torch.nn.Module):
    """Raise KeyboardInterrupt during model execution for lifecycle tests."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.1))

    def forward(self, representations: torch.Tensor, position_mask: torch.Tensor) -> torch.Tensor:
        del representations, position_mask
        raise KeyboardInterrupt


def _fold_size_score(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Return the held-out fold size for deterministic CV assertions."""
    del targets
    return float(predictions.numel())


def test_leaf_tracking_coordinator_selects_injected_tracker_and_records_lifecycle(capsys):
    """Private lifecycle state sanitizes payloads and locks one terminal state."""
    from phylognn.leaf_regression.tracking import _LeafExperimentCoordinator
    from phylognn.training.tracking import TrackingConfig

    tracker = _ObservableLeafTracker()
    coordinator = _LeafExperimentCoordinator(
        TrackingConfig(enabled=True, project="phylognn"), tracker=tracker
    )

    coordinator.start({"workflow.type": "fit", "data.config_file": "/private/input.toml"})
    fit_stage = coordinator.start_stage("fit")
    coordinator.log_epoch(fit_stage, loss=0.5, learning_rate=0.01, epoch_time_sec=0.2)
    cv_stage = coordinator.start_stage("cv_fold")
    coordinator.log_epoch(cv_stage, loss=0.25, learning_rate=0.01, epoch_time_sec=0.3)
    coordinator.finish("completed")
    coordinator.finish("failed")

    assert tracker.start_payloads == [{"data.config_file": "input.toml", "workflow.type": "fit"}]
    assert [step for step, _ in tracker.metric_calls] == [1, 2, 2]
    assert tracker.metric_calls[0][1] == {
        "epoch_time_sec": 0.2,
        "lr": 0.01,
        "stage/epoch": 1,
        "stage/index": 1,
        "stage/type": "fit",
        "train/loss": 0.5,
    }
    assert tracker.metric_calls[1][1]["stage/type"] == "cv_fold"
    assert tracker.metric_calls[-1] == (2, {"status/state": "completed"})
    assert tracker.finish_calls == ["completed"]
    assert capsys.readouterr().out.strip() == (
        "Tracking run: id=leaf-run name=leaf-name url=https://test/run"
    )


def test_leaf_tracking_coordinator_keeps_disabled_and_backend_selection_inert(monkeypatch):
    """Only an enabled configuration selects a backend or calls an injected tracker."""
    from phylognn.leaf_regression.tracking import _LeafExperimentCoordinator
    from phylognn.training.tracking import TrackingConfig

    disabled_tracker = _ObservableLeafTracker()
    disabled = _LeafExperimentCoordinator(TrackingConfig(enabled=False), tracker=disabled_tracker)
    disabled.start({"workflow.type": "fit"})
    disabled.log_epoch(disabled.start_stage("fit"), loss=1.0, learning_rate=0.1, epoch_time_sec=0.0)
    disabled.finish("completed")
    assert disabled_tracker.start_payloads == []
    assert disabled_tracker.metric_calls == []
    assert disabled_tracker.finish_calls == []

    created_tracker = _ObservableLeafTracker()
    monkeypatch.setattr(
        "phylognn.leaf_regression.tracking.create_tracker",
        lambda config: created_tracker,
    )
    selected = _LeafExperimentCoordinator(TrackingConfig(enabled=True, project="phylognn"))
    selected.start({"workflow.type": "fit"})

    assert created_tracker.start_payloads == [{"workflow.type": "fit"}]


@pytest.mark.parametrize("entry_point", ["fit", "cross_validate", "run"])
def test_disabled_leaf_tracking_never_calls_injected_tracker_or_imports_backend(
    monkeypatch,
    entry_point,
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
):
    """Disabled tracking stays inert for every public leaf-regression entry point."""
    from phylognn import (
        LeafRegressionConfig,
        cross_validate_leaf_regression,
        fit_leaf_regression,
        prepare_leaf_regression,
        run_leaf_regression,
    )
    from phylognn.training import TrackingConfig

    def fail_import(name):
        raise AssertionError(f"optional backend imported while tracking is disabled: {name}")

    monkeypatch.setattr("phylognn.training.tracking.import_module", fail_import)
    tracker = _ObservableLeafTracker()
    config = LeafRegressionConfig(epochs=1, learning_rate=0.01, seed=19)
    model_kwargs = {"model_class": _ConfiguredRegressor, "model_config": {"offset": 0.2}}

    if entry_point == "fit":
        data = prepare_leaf_regression(
            leaf_regression_tree,
            leaf_regression_representations,
            leaf_regression_position_mask,
            leaf_regression_targets,
        )
        fit_leaf_regression(
            data,
            training_config=config,
            tracking_config=TrackingConfig(enabled=False),
            tracker=tracker,
            **model_kwargs,
        )
    elif entry_point == "cross_validate":
        data = prepare_leaf_regression(
            leaf_regression_tree,
            leaf_regression_representations,
            leaf_regression_position_mask,
            leaf_regression_targets,
        )
        cross_validate_leaf_regression(
            data,
            validation_folds=([0, 1], [2, 3], [4, 5]),
            training_config=config,
            score_fn=_fold_size_score,
            refit=False,
            tracking_config=TrackingConfig(enabled=False),
            tracker=tracker,
            **model_kwargs,
        )
    else:
        run_leaf_regression(
            leaf_regression_tree,
            leaf_regression_representations,
            leaf_regression_position_mask,
            leaf_regression_targets,
            validation_folds=([0, 1], [2, 3], [4, 5]),
            training_config=config,
            score_fn=_fold_size_score,
            tracking_config=TrackingConfig(enabled=False),
            tracker=tracker,
            **model_kwargs,
        )

    assert tracker.start_payloads == []
    assert tracker.metric_calls == []
    assert tracker.finish_calls == []


def test_run_leaf_regression_records_three_folds_and_refit_in_one_run(
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
):
    """A tracked three-fold workflow records every epoch with stable stage fields."""
    from phylognn import LeafRegressionConfig, run_leaf_regression
    from phylognn.training.tracking import TrackingConfig

    tracker = _ObservableLeafTracker()
    result = run_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
        validation_folds=([0, 1], [2, 3], [4, 5]),
        training_config=LeafRegressionConfig(epochs=20, learning_rate=0.01, seed=11),
        score_fn=_fold_size_score,
        model_class=_ConfiguredRegressor,
        model_config={"offset": 0.2},
        tracking_config=TrackingConfig(enabled=True, project="phylognn"),
        tracker=tracker,
    )

    epoch_calls = [
        (step, metrics) for step, metrics in tracker.metric_calls if "train/loss" in metrics
    ]
    assert len(epoch_calls) == 80
    assert [step for step, _ in epoch_calls] == list(range(1, 81))
    assert {(metrics["stage/type"], metrics["stage/index"]) for _, metrics in epoch_calls} == {
        ("cv_fold", 1),
        ("cv_fold", 2),
        ("cv_fold", 3),
        ("refit", 1),
    }
    for stage_type, stage_index in (
        ("cv_fold", 1),
        ("cv_fold", 2),
        ("cv_fold", 3),
        ("refit", 1),
    ):
        epochs = [
            metrics["stage/epoch"]
            for _, metrics in epoch_calls
            if metrics["stage/type"] == stage_type and metrics["stage/index"] == stage_index
        ]
        assert epochs == list(range(1, 21))
    summaries = [metrics for _, metrics in tracker.metric_calls if "cv/fold_score" in metrics]
    assert len(summaries) == 3
    weighted = [metrics for _, metrics in tracker.metric_calls if "cv/weighted_score" in metrics]
    assert len(weighted) == 1
    assert tracker.metric_calls[-1][1] == {"status/state": "completed"}
    assert tracker.finish_calls == ["completed"]
    assert result.cv_score == pytest.approx(2.0)


def test_run_leaf_regression_has_one_tracker_start_and_identity_output(
    capsys,
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
):
    """The workflow owns one tracker run and exposes its identity locally."""
    from phylognn import LeafRegressionConfig, run_leaf_regression
    from phylognn.training.tracking import TrackingConfig

    tracker = _ObservableLeafTracker()
    run_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
        validation_folds=([0, 1], [2, 3], [4, 5]),
        training_config=LeafRegressionConfig(epochs=1, seed=13),
        score_fn=_fold_size_score,
        model_class=_ConfiguredRegressor,
        model_config={"offset": 0.2},
        tracking_config=TrackingConfig(enabled=True, project="phylognn"),
        tracker=tracker,
    )

    assert len(tracker.start_payloads) == 1
    assert tracker.start_payloads[0]["workflow.type"] == "run"
    assert tracker.start_payloads[0]["data.leaf_count"] == 6
    assert tracker.start_payloads[0]["cv.fold_count"] == 3
    assert tracker.start_payloads[0]["cv.refit"] is True
    assert tracker.start_payloads[0]["cv.score"] == "_fold_size_score"
    assert not any("child" in str(payload).lower() for payload in tracker.start_payloads)
    payloads = [*tracker.start_payloads, *(metrics for _, metrics in tracker.metric_calls)]
    assert all(
        value is not leaf_regression_tree
        and value is not leaf_regression_representations
        and value is not leaf_regression_position_mask
        and value is not leaf_regression_targets
        and not isinstance(value, (torch.Tensor, torch.nn.Module))
        for payload in payloads
        for value in payload.values()
    )
    assert all(
        not any("leaf_name" in key or "leaf_index" in key for key in payload)
        for payload in payloads
    )
    assert capsys.readouterr().out.strip() == (
        "Tracking run: id=leaf-run name=leaf-name url=https://test/run"
    )


def test_run_leaf_regression_records_failure_interruption_and_cleanup_warning(
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
):
    """Workflow errors preserve the primary exception and never duplicate terminal states."""
    from phylognn import LeafRegressionConfig, run_leaf_regression
    from phylognn.training.tracking import TrackingConfig, TrackingError

    config = TrackingConfig(enabled=True, project="phylognn")
    failed_tracker = _ObservableLeafTracker(fail_finish=True)
    with pytest.raises(RuntimeError, match="construction failed"), pytest.warns(
        UserWarning, match="Tracking cleanup failed after training failed"
    ):
        run_leaf_regression(
            leaf_regression_tree,
            leaf_regression_representations,
            leaf_regression_position_mask,
            leaf_regression_targets,
            validation_folds=([0, 1], [2, 3], [4, 5]),
            training_config=LeafRegressionConfig(epochs=1, seed=17),
            score_fn=_fold_size_score,
            model_class=_ConstructorFailureRegressor,
            tracking_config=config,
            tracker=failed_tracker,
        )
    statuses = [
        metrics["status/state"]
        for _, metrics in failed_tracker.metric_calls
        if "status/state" in metrics
    ]
    assert statuses == ["failed"]
    assert failed_tracker.finish_calls == ["failed"]

    interrupted_tracker = _ObservableLeafTracker()
    with pytest.raises(KeyboardInterrupt):
        run_leaf_regression(
            leaf_regression_tree,
            leaf_regression_representations,
            leaf_regression_position_mask,
            leaf_regression_targets,
            validation_folds=([0, 1], [2, 3], [4, 5]),
            training_config=LeafRegressionConfig(epochs=1, seed=17),
            score_fn=_fold_size_score,
            model_class=_InterruptingRegressor,
            tracking_config=config,
            tracker=interrupted_tracker,
        )
    statuses = [
        metrics["status/state"]
        for _, metrics in interrupted_tracker.metric_calls
        if "status/state" in metrics
    ]
    assert statuses == ["interrupted"]

    logging_tracker = _ObservableLeafTracker(fail_metric_key="cv/weighted_score")
    with pytest.raises(TrackingError, match="metric failure"):
        run_leaf_regression(
            leaf_regression_tree,
            leaf_regression_representations,
            leaf_regression_position_mask,
            leaf_regression_targets,
            validation_folds=([0, 1], [2, 3], [4, 5]),
            training_config=LeafRegressionConfig(epochs=1, seed=17),
            score_fn=_fold_size_score,
            model_class=_ConfiguredRegressor,
            model_config={"offset": 0.2},
            tracking_config=config,
            tracker=logging_tracker,
        )
    statuses = [
        metrics["status/state"]
        for _, metrics in logging_tracker.metric_calls
        if "status/state" in metrics
    ]
    assert statuses == ["failed"]


def test_fit_leaf_regression_tracking_records_direct_stage_and_lifecycle(
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
):
    """A direct fit owns one tracked ``fit:1`` experiment."""
    from phylognn import LeafRegressionConfig, fit_leaf_regression, prepare_leaf_regression
    from phylognn.training.tracking import TrackingConfig

    data = prepare_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
    )
    tracker = _ObservableLeafTracker()
    result = fit_leaf_regression(
        data,
        train_indices=[0, 2, 4],
        training_config=LeafRegressionConfig(epochs=2, learning_rate=0.01, seed=31),
        model_class=_ConfiguredRegressor,
        model_config={"offset": 0.2},
        tracking_config=TrackingConfig(enabled=True, project="phylognn"),
        tracker=tracker,
    )

    assert len(result.losses) == 2
    assert {
        "workflow.type": "fit",
        "data.leaf_count": 6,
        "cv.fold_count": 0,
        "cv.refit": False,
        "cv.score": "r2",
        "training.epochs": 2,
        "training.learning_rate": 0.01,
        "training.weight_decay": 0.0,
        "training.seed": 31,
        "training.device": "cpu",
    }.items() <= tracker.start_payloads[0].items()
    epoch_calls = [call for call in tracker.metric_calls if "train/loss" in call[1]]
    assert [step for step, _ in epoch_calls] == [1, 2]
    for index, (_, metrics) in enumerate(epoch_calls, start=1):
        assert metrics["stage/type"] == "fit"
        assert metrics["stage/index"] == 1
        assert metrics["stage/epoch"] == index
        assert np.isfinite(metrics["train/loss"])
        assert metrics["lr"] == 0.01
        assert np.isfinite(metrics["epoch_time_sec"])
    assert tracker.metric_calls[-1] == (2, {"status/state": "completed"})
    assert tracker.finish_calls == ["completed"]


def test_fit_leaf_regression_tracking_preserves_failure_and_cleanup_precedence(
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
):
    """Direct-fit failures retain their primary error and one terminal status."""
    from phylognn import LeafRegressionConfig, fit_leaf_regression, prepare_leaf_regression
    from phylognn.training.tracking import TrackingConfig, TrackingError

    data = prepare_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
    )
    config = TrackingConfig(enabled=True, project="phylognn")
    cleanup_tracker = _ObservableLeafTracker(fail_finish=True)
    with pytest.raises(RuntimeError, match="construction failed"), pytest.warns(
        UserWarning, match="Tracking cleanup failed after training failed"
    ):
        fit_leaf_regression(
            data,
            training_config=LeafRegressionConfig(epochs=1, seed=37),
            model_class=_ConstructorFailureRegressor,
            tracking_config=config,
            tracker=cleanup_tracker,
        )
    assert cleanup_tracker.finish_calls == ["failed"]
    assert [
        metrics["status/state"]
        for _, metrics in cleanup_tracker.metric_calls
        if "status/state" in metrics
    ] == ["failed"]

    interrupted_tracker = _ObservableLeafTracker()
    with pytest.raises(KeyboardInterrupt):
        fit_leaf_regression(
            data,
            training_config=LeafRegressionConfig(epochs=1, seed=37),
            model_class=_InterruptingRegressor,
            tracking_config=config,
            tracker=interrupted_tracker,
        )
    assert [
        metrics["status/state"]
        for _, metrics in interrupted_tracker.metric_calls
        if "status/state" in metrics
    ] == ["interrupted"]

    logging_tracker = _ObservableLeafTracker(fail_metric_key="train/loss")
    with pytest.raises(TrackingError, match="metric failure"):
        fit_leaf_regression(
            data,
            training_config=LeafRegressionConfig(epochs=1, seed=37),
            model_class=_ConfiguredRegressor,
            model_config={"offset": 0.2},
            tracking_config=config,
            tracker=logging_tracker,
        )
    assert [
        metrics["status/state"]
        for _, metrics in logging_tracker.metric_calls
        if "status/state" in metrics
    ] == ["failed"]


def test_cross_validate_leaf_regression_tracking_records_direct_stages_and_summaries(
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
):
    """Direct CV records ordered folds, summaries, and no refit when disabled."""
    from phylognn import (
        LeafRegressionConfig,
        cross_validate_leaf_regression,
        prepare_leaf_regression,
    )
    from phylognn.training.tracking import TrackingConfig

    data = prepare_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
    )
    tracker = _ObservableLeafTracker()
    result = cross_validate_leaf_regression(
        data,
        validation_folds=([0, 1], [2, 3], [4, 5]),
        training_config=LeafRegressionConfig(epochs=2, learning_rate=0.01, seed=41),
        score_fn=_fold_size_score,
        refit=False,
        model_class=_ConfiguredRegressor,
        model_config={"offset": 0.2},
        tracking_config=TrackingConfig(enabled=True, project="phylognn"),
        tracker=tracker,
    )

    assert tracker.start_payloads[0]["workflow.type"] == "cross_validate"
    assert tracker.start_payloads[0]["data.leaf_count"] == 6
    assert tracker.start_payloads[0]["cv.fold_count"] == 3
    assert tracker.start_payloads[0]["cv.refit"] is False
    assert tracker.start_payloads[0]["cv.score"] == "_fold_size_score"
    epoch_calls = [call for call in tracker.metric_calls if "train/loss" in call[1]]
    assert [step for step, _ in epoch_calls] == list(range(1, 7))
    assert [
        (metrics["stage/type"], metrics["stage/index"], metrics["stage/epoch"])
        for _, metrics in epoch_calls
    ] == [
        ("cv_fold", 1, 1),
        ("cv_fold", 1, 2),
        ("cv_fold", 2, 1),
        ("cv_fold", 2, 2),
        ("cv_fold", 3, 1),
        ("cv_fold", 3, 2),
    ]
    summaries = [metrics for _, metrics in tracker.metric_calls if "cv/fold_score" in metrics]
    assert [summary["stage/index"] for summary in summaries] == [1, 2, 3]
    assert [summary["cv/validation_leaf_count"] for summary in summaries] == [2, 2, 2]
    assert [summary["cv/fold_score"] for summary in summaries] == [2.0, 2.0, 2.0]
    assert [
        metrics["cv/weighted_score"]
        for _, metrics in tracker.metric_calls
        if "cv/weighted_score" in metrics
    ] == [2.0]
    assert result.cv_score == pytest.approx(2.0)
    assert result.final_fit is None
    assert not any(metrics.get("stage/type") == "refit" for _, metrics in tracker.metric_calls)
    assert tracker.metric_calls[-1] == (6, {"status/state": "completed"})
    assert tracker.finish_calls == ["completed"]


def test_cross_validate_leaf_regression_tracking_preserves_failure_and_cleanup_precedence(
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
):
    """Direct-CV failures retain their primary error and one terminal status."""
    from phylognn import (
        LeafRegressionConfig,
        cross_validate_leaf_regression,
        prepare_leaf_regression,
    )
    from phylognn.training.tracking import TrackingConfig, TrackingError

    data = prepare_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
    )
    kwargs = {
        "validation_folds": ([0, 1], [2, 3], [4, 5]),
        "training_config": LeafRegressionConfig(epochs=1, seed=43),
        "score_fn": _fold_size_score,
        "tracking_config": TrackingConfig(enabled=True, project="phylognn"),
    }
    cleanup_tracker = _ObservableLeafTracker(fail_finish=True)
    with pytest.raises(RuntimeError, match="construction failed"), pytest.warns(
        UserWarning, match="Tracking cleanup failed after training failed"
    ):
        cross_validate_leaf_regression(
            data,
            model_class=_ConstructorFailureRegressor,
            tracker=cleanup_tracker,
            **kwargs,
        )
    assert cleanup_tracker.finish_calls == ["failed"]
    assert [
        metrics["status/state"]
        for _, metrics in cleanup_tracker.metric_calls
        if "status/state" in metrics
    ] == ["failed"]

    interrupted_tracker = _ObservableLeafTracker()
    with pytest.raises(KeyboardInterrupt):
        cross_validate_leaf_regression(
            data,
            model_class=_InterruptingRegressor,
            tracker=interrupted_tracker,
            **kwargs,
        )
    assert [
        metrics["status/state"]
        for _, metrics in interrupted_tracker.metric_calls
        if "status/state" in metrics
    ] == ["interrupted"]

    logging_tracker = _ObservableLeafTracker(fail_metric_key="cv/weighted_score")
    with pytest.raises(TrackingError, match="metric failure"):
        cross_validate_leaf_regression(
            data,
            model_class=_ConfiguredRegressor,
            model_config={"offset": 0.2},
            tracker=logging_tracker,
            **kwargs,
        )
    assert [
        metrics["status/state"]
        for _, metrics in logging_tracker.metric_calls
        if "status/state" in metrics
    ] == ["failed"]


def test_direct_tracking_preserves_fit_and_cv_scientific_results(
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
):
    """Tracking does not change direct-fit or CV results, including no-attention models."""
    from phylognn import (
        LeafRegressionConfig,
        cross_validate_leaf_regression,
        fit_leaf_regression,
        prepare_leaf_regression,
    )
    from phylognn.training.tracking import TrackingConfig

    data = prepare_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
    )
    config = LeafRegressionConfig(epochs=2, learning_rate=0.01, seed=43)
    model_kwargs = {"model_class": _PredictionOnlyRegressor, "model_config": {"offset": 0.2}}
    tracked_config = TrackingConfig(enabled=True, project="phylognn")

    fit_without_tracking = fit_leaf_regression(data, training_config=config, **model_kwargs)
    fit_with_tracking = fit_leaf_regression(
        data,
        training_config=config,
        tracking_config=tracked_config,
        tracker=_ObservableLeafTracker(),
        **model_kwargs,
    )
    cv_without_tracking = cross_validate_leaf_regression(
        data,
        validation_folds=([0, 1], [2, 3], [4, 5]),
        training_config=config,
        score_fn=_fold_size_score,
        **model_kwargs,
    )
    cv_with_tracking = cross_validate_leaf_regression(
        data,
        validation_folds=([0, 1], [2, 3], [4, 5]),
        training_config=config,
        score_fn=_fold_size_score,
        tracking_config=tracked_config,
        tracker=_ObservableLeafTracker(),
        **model_kwargs,
    )

    assert fit_with_tracking.losses == fit_without_tracking.losses
    assert torch.equal(fit_with_tracking.predictions, fit_without_tracking.predictions)
    assert fit_with_tracking.attention is None
    assert cv_with_tracking.fold_scores == cv_without_tracking.fold_scores
    assert cv_with_tracking.cv_score == cv_without_tracking.cv_score
    assert torch.equal(cv_with_tracking.oof_predictions, cv_without_tracking.oof_predictions)
    assert tuple(result.losses for result in cv_with_tracking.fold_results) == tuple(
        result.losses for result in cv_without_tracking.fold_results
    )
    assert all(result.attention is None for result in cv_with_tracking.fold_results)

    attention_kwargs = {
        "model_class": _ConfiguredRegressor,
        "model_config": {"offset": 0.2},
    }
    fit_without_attention_tracking = fit_leaf_regression(
        data, training_config=config, **attention_kwargs
    )
    fit_with_attention_tracking = fit_leaf_regression(
        data,
        training_config=config,
        tracking_config=tracked_config,
        tracker=_ObservableLeafTracker(),
        **attention_kwargs,
    )
    cv_without_attention_tracking = cross_validate_leaf_regression(
        data,
        validation_folds=([0, 1], [2, 3], [4, 5]),
        training_config=config,
        score_fn=_fold_size_score,
        **attention_kwargs,
    )
    cv_with_attention_tracking = cross_validate_leaf_regression(
        data,
        validation_folds=([0, 1], [2, 3], [4, 5]),
        training_config=config,
        score_fn=_fold_size_score,
        tracking_config=tracked_config,
        tracker=_ObservableLeafTracker(),
        **attention_kwargs,
    )

    assert torch.equal(
        fit_with_attention_tracking.predictions, fit_without_attention_tracking.predictions
    )
    assert torch.equal(
        fit_with_attention_tracking.attention, fit_without_attention_tracking.attention
    )
    assert all(
        torch.equal(tracked.attention, untracked.attention)
        for tracked, untracked in zip(
            cv_with_attention_tracking.fold_results,
            cv_without_attention_tracking.fold_results,
            strict=True,
        )
    )
    assert cv_with_attention_tracking.final_fit is not None
    assert cv_without_attention_tracking.final_fit is not None
    assert torch.equal(
        cv_with_attention_tracking.final_fit.attention,
        cv_without_attention_tracking.final_fit.attention,
    )


def test_default_leaf_regression_calls_preserve_results_rng_and_files(
    tmp_path,
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
):
    """Untracked public calls retain result fields, caller RNG, and no files."""
    from phylognn import (
        LeafRegressionConfig,
        cross_validate_leaf_regression,
        fit_leaf_regression,
        prepare_leaf_regression,
        run_leaf_regression,
    )

    data = prepare_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
    )
    config = LeafRegressionConfig(epochs=1, learning_rate=0.01, seed=29)
    model_kwargs = {"model_class": _PredictionOnlyRegressor, "model_config": {"offset": 0.2}}
    random.seed(211)
    np.random.seed(211)
    torch.manual_seed(211)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()

    fit_result = fit_leaf_regression(data, training_config=config, **model_kwargs)
    cv_result = cross_validate_leaf_regression(
        data,
        validation_folds=([0, 1], [2, 3], [4, 5]),
        training_config=config,
        score_fn=_fold_size_score,
        refit=False,
        **model_kwargs,
    )
    run_result = run_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
        validation_folds=([0, 1], [2, 3], [4, 5]),
        training_config=config,
        score_fn=_fold_size_score,
        **model_kwargs,
    )

    assert set(fit_result.__dataclass_fields__) == {
        "predictions",
        "attention",
        "train_indices",
        "losses",
    }
    assert set(cv_result.__dataclass_fields__) == {
        "cv_score",
        "fold_scores",
        "oof_predictions",
        "validation_folds",
        "fold_results",
        "final_fit",
    }
    assert set(run_result.__dataclass_fields__) == {
        "cv_score",
        "fold_scores",
        "oof_predictions",
        "predictions",
        "attention",
        "mean_attention",
    }
    assert fit_result.attention is None
    assert all(result.attention is None for result in cv_result.fold_results)
    assert run_result.attention is None
    assert run_result.mean_attention is None
    assert random.getstate() == python_state
    actual_numpy_state = np.random.get_state()
    assert actual_numpy_state[0] == numpy_state[0]
    assert np.array_equal(actual_numpy_state[1], numpy_state[1])
    assert actual_numpy_state[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert list(tmp_path.iterdir()) == []


def test_run_leaf_regression_uses_default_model_and_returns_complete_results(
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
):
    """The default model completes CV and one final refit without file output."""
    from phylognn import LeafRegressionConfig, run_leaf_regression

    result = run_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
        n_splits=3,
        training_config=LeafRegressionConfig(epochs=2, learning_rate=0.01, seed=17),
    )

    assert len(result.fold_scores) == 3
    assert torch.isfinite(result.oof_predictions).all()
    assert torch.isfinite(result.predictions).all()
    assert result.oof_predictions.shape == leaf_regression_targets.shape
    assert result.attention is not None
    assert result.mean_attention is not None


def test_run_leaf_regression_constructs_fresh_configured_models_and_weights_scores(
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
):
    """Every fold and refit receives a new class-plus-keyword model instance."""
    from phylognn import LeafRegressionConfig, run_leaf_regression

    _ConfiguredRegressor.instances.clear()
    folds = ([0, 1, 2], [3, 4], [5])
    result = run_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
        validation_folds=folds,
        training_config=LeafRegressionConfig(epochs=1, seed=7),
        score_fn=lambda predictions, targets: float(targets.numel()),
        model_class=_ConfiguredRegressor,
        model_config={"offset": 0.5},
    )

    assert len(_ConfiguredRegressor.instances) == len(folds) + 1
    assert len({id(model) for model in _ConfiguredRegressor.instances}) == len(folds) + 1
    assert all(model.offset == 0.5 for model in _ConfiguredRegressor.instances)
    assert result.fold_scores == (3.0, 2.0, 1.0)
    assert result.cv_score == pytest.approx((3.0 * 3 + 2.0 * 2 + 1.0) / 6)
    assert torch.isfinite(result.oof_predictions).all()


def test_run_leaf_regression_is_repeatable_and_restores_caller_rng_states(
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
):
    """A successful seeded workflow leaves Python, NumPy, and PyTorch RNG untouched."""
    from phylognn import LeafRegressionConfig, run_leaf_regression

    kwargs = {
        "n_splits": 3,
        "training_config": LeafRegressionConfig(epochs=1, seed=23),
    }
    random.seed(101)
    np.random.seed(101)
    torch.manual_seed(101)
    expected_python = random.random()
    expected_numpy = np.random.rand()
    expected_torch = torch.rand(3)
    random.seed(101)
    np.random.seed(101)
    torch.manual_seed(101)

    first = run_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
        **kwargs,
    )
    actual_python = random.random()
    actual_numpy = np.random.rand()
    actual_torch = torch.rand(3)
    second = run_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
        **kwargs,
    )

    assert actual_python == expected_python
    assert actual_numpy == expected_numpy
    assert torch.equal(actual_torch, expected_torch)
    assert first.fold_scores == second.fold_scores
    assert torch.equal(first.oof_predictions, second.oof_predictions)
    assert torch.equal(first.predictions, second.predictions)


def test_prepare_leaf_regression_preserves_rows_and_aligns_mapped_targets(
    leaf_regression_permuted_leaf_names,
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_target_mapping,
    leaf_regression_targets,
    leaf_regression_tree,
):
    """Explicit leaf order aligns mapped targets without reordering caller rows."""
    from phylognn import prepare_leaf_regression

    representations = leaf_regression_representations.clone()
    position_mask = leaf_regression_position_mask.clone()
    data = prepare_leaf_regression(
        leaf_regression_tree,
        representations,
        position_mask,
        leaf_regression_target_mapping,
        leaf_names=leaf_regression_permuted_leaf_names,
    )

    expected_targets = torch.tensor(
        [leaf_regression_target_mapping[name] for name in leaf_regression_permuted_leaf_names],
        dtype=torch.float32,
    )
    assert data.leaf_names == leaf_regression_permuted_leaf_names
    assert torch.equal(data.targets, expected_targets)
    assert torch.equal(data.representations, representations)
    assert torch.equal(data.position_mask, position_mask)
    assert torch.equal(representations, leaf_regression_representations)
    assert torch.equal(position_mask, leaf_regression_position_mask)
    assert not torch.equal(data.targets, leaf_regression_targets)


def test_fit_leaf_regression_returns_selected_indices_all_predictions_and_losses(
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
):
    """A staged fit trains selected leaves yet evaluates every prepared leaf."""
    from phylognn import LeafRegressionConfig, fit_leaf_regression, prepare_leaf_regression

    data = prepare_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
    )
    selected_indices = [5, 1, 3]
    result = fit_leaf_regression(
        data,
        train_indices=selected_indices,
        training_config=LeafRegressionConfig(epochs=3, learning_rate=0.01, seed=19),
        model_class=_ConfiguredRegressor,
        model_config={"offset": 0.25},
    )

    assert torch.equal(result.train_indices, torch.tensor(selected_indices))
    assert result.predictions.shape == (len(data.leaf_names),)
    assert result.attention is not None
    assert result.attention.shape == data.position_mask.shape
    assert len(result.losses) == 3
    assert all(np.isfinite(loss) for loss in result.losses)


def test_prediction_only_model_propagates_none_attention_across_workflows(
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
):
    """Prediction-only models never receive synthesized attention values."""
    from phylognn import (
        LeafRegressionConfig,
        cross_validate_leaf_regression,
        fit_leaf_regression,
        prepare_leaf_regression,
        run_leaf_regression,
    )

    config = LeafRegressionConfig(epochs=1, learning_rate=0.01, seed=37)
    model_kwargs = {"model_class": _PredictionOnlyRegressor, "model_config": {"offset": 0.2}}
    data = prepare_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
    )

    fit_result = fit_leaf_regression(data, training_config=config, **model_kwargs)
    cv_result = cross_validate_leaf_regression(
        data,
        validation_folds=([0, 1], [2, 3], [4, 5]),
        training_config=config,
        score_fn=_fold_size_score,
        **model_kwargs,
    )
    workflow_result = run_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
        validation_folds=([0, 1], [2, 3], [4, 5]),
        training_config=config,
        score_fn=_fold_size_score,
        **model_kwargs,
    )

    assert fit_result.attention is None
    assert all(result.attention is None for result in cv_result.fold_results)
    assert cv_result.final_fit is not None
    assert cv_result.final_fit.attention is None
    assert workflow_result.attention is None
    assert workflow_result.mean_attention is None


def test_attention_model_preserves_masked_attention_across_workflows(
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
):
    """Attention-producing models retain real masked attention at every stage."""
    from phylognn import (
        LeafRegressionConfig,
        cross_validate_leaf_regression,
        fit_leaf_regression,
        prepare_leaf_regression,
        run_leaf_regression,
    )

    config = LeafRegressionConfig(epochs=1, learning_rate=0.01, seed=41)
    model_kwargs = {"model_class": _ConfiguredRegressor, "model_config": {"offset": 0.2}}
    data = prepare_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
    )

    fit_result = fit_leaf_regression(data, training_config=config, **model_kwargs)
    cv_result = cross_validate_leaf_regression(
        data,
        validation_folds=([0, 1], [2, 3], [4, 5]),
        training_config=config,
        score_fn=_fold_size_score,
        **model_kwargs,
    )
    workflow_result = run_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
        validation_folds=([0, 1], [2, 3], [4, 5]),
        training_config=config,
        score_fn=_fold_size_score,
        **model_kwargs,
    )

    assert fit_result.attention is not None
    assert fit_result.attention.shape == data.position_mask.shape
    assert torch.all(fit_result.attention[~data.position_mask] == 0)
    assert all(result.attention is not None for result in cv_result.fold_results)
    assert cv_result.final_fit is not None
    assert cv_result.final_fit.attention is not None
    assert workflow_result.attention is not None
    assert workflow_result.mean_attention is not None
    assert workflow_result.mean_attention.shape == (data.representations.size(1),)
    assert torch.all(workflow_result.attention[~data.position_mask] == 0)


def test_cross_validation_preserves_manual_folds_and_matches_recommended_workflow(
    leaf_regression_permuted_leaf_names,
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_target_mapping,
    leaf_regression_tree,
):
    """Manual folds retain caller order, cover OOF output, and match the facade."""
    from phylognn import (
        LeafRegressionConfig,
        cross_validate_leaf_regression,
        prepare_leaf_regression,
        run_leaf_regression,
    )

    folds = ([5], [2, 0], [4, 3, 1])
    config = LeafRegressionConfig(epochs=2, learning_rate=0.01, seed=29)
    model_config = {"offset": 0.75}
    data = prepare_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_target_mapping,
        leaf_names=leaf_regression_permuted_leaf_names,
    )

    without_refit = cross_validate_leaf_regression(
        data,
        n_splits=99,
        validation_folds=folds,
        training_config=config,
        score_fn=_fold_size_score,
        refit=False,
        model_class=_ConfiguredRegressor,
        model_config=model_config,
    )
    with_refit = cross_validate_leaf_regression(
        data,
        n_splits=99,
        validation_folds=folds,
        training_config=config,
        score_fn=_fold_size_score,
        model_class=_ConfiguredRegressor,
        model_config=model_config,
    )
    recommended = run_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_target_mapping,
        leaf_names=leaf_regression_permuted_leaf_names,
        n_splits=99,
        validation_folds=folds,
        training_config=config,
        score_fn=_fold_size_score,
        model_class=_ConfiguredRegressor,
        model_config=model_config,
    )

    assert tuple(fold.tolist() for fold in without_refit.validation_folds) == tuple(folds)
    assert torch.isfinite(without_refit.oof_predictions).all()
    assert torch.equal(
        torch.sort(torch.cat(without_refit.validation_folds)).values,
        torch.arange(len(data.leaf_names), dtype=torch.long),
    )
    assert without_refit.final_fit is None
    assert with_refit.final_fit is not None
    assert torch.equal(
        with_refit.final_fit.train_indices,
        torch.arange(len(data.leaf_names), dtype=torch.long),
    )
    assert torch.equal(with_refit.oof_predictions, recommended.oof_predictions)
    assert with_refit.fold_scores == recommended.fold_scores
    assert with_refit.cv_score == recommended.cv_score
    assert torch.equal(with_refit.final_fit.predictions, recommended.predictions)


@pytest.mark.parametrize(
    ("representations", "position_mask", "targets", "leaf_names", "error"),
    [
        (torch.ones((6, 4), dtype=torch.float32), None, None, None, "representations"),
        (torch.ones((6, 4, 3), dtype=torch.int64), None, None, None, "representations"),
        (torch.full((6, 4, 3), float("nan")), None, None, None, "representations"),
        (None, torch.ones((6, 4), dtype=torch.float32) * 2, None, None, "position_mask"),
        (None, torch.tensor([[1.0, float("nan"), 0.0, 0.0]] * 6), None, None, "position_mask"),
        (None, torch.zeros((6, 4), dtype=torch.int64), None, None, "position_mask"),
        (None, None, torch.ones((6, 1), dtype=torch.float32), None, "targets"),
        (None, None, torch.full((6,), float("inf")), None, "targets"),
        (None, None, {"A": 1.0}, None, "Target mapping"),
        (None, None, None, ("A", "B", "C", "D", "E", "missing"), "leaf_names"),
    ],
)
def test_prepare_leaf_regression_rejects_invalid_alignment_contracts(
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
    representations,
    position_mask,
    targets,
    leaf_names,
    error,
):
    """Preparation rejects invalid shapes, values, masks, targets, and leaf permutations."""
    from phylognn import prepare_leaf_regression

    with pytest.raises((TypeError, ValueError), match=error):
        prepare_leaf_regression(
            leaf_regression_tree,
            leaf_regression_representations if representations is None else representations,
            leaf_regression_position_mask if position_mask is None else position_mask,
            leaf_regression_targets if targets is None else targets,
            leaf_names=leaf_names,
        )


def test_prepare_leaf_regression_canonicalizes_strict_numeric_inputs_and_supports_one_leaf(
    ete3_module,
):
    """Preparation accepts only numeric zero/one masks and canonicalizes accepted data."""
    from phylognn import prepare_leaf_regression

    tree = ete3_module.Tree("A;")
    data = prepare_leaf_regression(
        tree,
        torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float64),
        [[1, 0]],
        torch.tensor([3.0], dtype=torch.float64),
    )

    assert data.representations.dtype == torch.float32
    assert data.targets.dtype == torch.float32
    assert data.position_mask.dtype == torch.bool
    assert data.position_mask.tolist() == [[True, False]]


def test_prepare_leaf_regression_rejects_invalid_tree_types_and_leaf_names(
    ete3_module,
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
):
    """Tree identity and leaf names are validated before tensor preparation."""
    from phylognn import prepare_leaf_regression

    with pytest.raises(TypeError, match="tree"):
        prepare_leaf_regression(
            object(),
            leaf_regression_representations,
            leaf_regression_position_mask,
            leaf_regression_targets,
        )
    with pytest.raises(ValueError, match="unique"):
        prepare_leaf_regression(
            ete3_module.Tree("(A:1,A:1,B:1);"),
            leaf_regression_representations[:3],
            leaf_regression_position_mask[:3],
            leaf_regression_targets[:3],
        )


@pytest.mark.parametrize(
    ("config", "model_class", "model_config", "error"),
    [
        (object(), None, None, "training_config"),
        (None, _NoParameterRegressor, {}, "trainable"),
        (None, _ConstructorFailureRegressor, {}, "construction failed"),
        (None, _InvalidOutputRegressor, {"kind": "shape"}, "predictions"),
        (None, _InvalidOutputRegressor, {"kind": "nonfinite"}, "predictions"),
        (None, _InvalidOutputRegressor, {"kind": "nondifferentiable"}, "loss"),
        (None, _InvalidOutputRegressor, {"kind": "attention"}, "attention"),
        (None, _InvalidOutputRegressor, {"kind": "tuple"}, "model must return"),
    ],
)
def test_fit_leaf_regression_rejects_invalid_pre_update_contracts(
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
    config,
    model_class,
    model_config,
    error,
):
    """Invalid fit contracts fail before an invalid model can update a parameter."""
    from phylognn import LeafRegressionConfig, fit_leaf_regression, prepare_leaf_regression

    _InvalidOutputRegressor.instances.clear()
    data = prepare_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
    )
    with pytest.raises((TypeError, ValueError, RuntimeError), match=error):
        fit_leaf_regression(
            data,
            train_indices=[0],
            training_config=config or LeafRegressionConfig(epochs=1, seed=11),
            model_class=model_class,
            model_config=model_config,
        )

    assert all(instance.weight.item() == 1.0 for instance in _InvalidOutputRegressor.instances)


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"epochs": 0}, "epochs"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"weight_decay": -1.0}, "weight_decay"),
        ({"seed": True}, "seed"),
        ({"device": object()}, "device"),
    ],
)
def test_leaf_regression_config_rejects_invalid_values(kwargs, error):
    """The training config accepts only its five validated control fields."""
    from phylognn import LeafRegressionConfig

    with pytest.raises(ValueError, match=error):
        LeafRegressionConfig(**kwargs)


def test_fit_leaf_regression_rejects_invalid_indices_and_model_definitions(
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
):
    """Index and model-definition errors are rejected before model training begins."""
    from phylognn import LeafRegressionConfig, fit_leaf_regression, prepare_leaf_regression

    data = prepare_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
    )
    config = LeafRegressionConfig(epochs=1, seed=17)
    for train_indices, model_class, model_config, error in (
        ([0, 0], _PredictionOnlyRegressor, {"offset": 0.0}, "duplicate"),
        ([6], _PredictionOnlyRegressor, {"offset": 0.0}, "outside"),
        ([0.0], _PredictionOnlyRegressor, {"offset": 0.0}, "integer"),
        ([0], object, {}, "model_class"),
        ([0], _PredictionOnlyRegressor, [], "model_config"),
    ):
        with pytest.raises((TypeError, ValueError), match=error):
            fit_leaf_regression(
                data,
                train_indices=train_indices,
                training_config=config,
                model_class=model_class,
                model_config=model_config,
            )
    with pytest.raises(TypeError, match="hidden_dim"):
        LeafRegressionConfig(hidden_dim=16)


def test_cross_validation_rejects_invalid_folds_and_scores_before_fitting(
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
):
    """Fold and custom-score failures occur before any fold model is constructed."""
    from phylognn import (
        LeafRegressionConfig,
        cross_validate_leaf_regression,
        prepare_leaf_regression,
    )

    data = prepare_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
    )
    _InvalidOutputRegressor.instances.clear()
    with pytest.raises(ValueError, match="validation_folds"):
        cross_validate_leaf_regression(data, validation_folds=([0, 1], [1, 2]))
    assert not _InvalidOutputRegressor.instances
    with pytest.raises(ValueError, match="score_fn"):
        cross_validate_leaf_regression(
            data,
            validation_folds=([0, 1], [2, 3], [4, 5]),
            training_config=LeafRegressionConfig(epochs=1, seed=13),
            score_fn=lambda predictions, targets: float("nan"),
            model_class=_InvalidOutputRegressor,
            model_config={"kind": "shape"},
        )
    assert not _InvalidOutputRegressor.instances


@pytest.mark.parametrize(
    "n_splits,error",
    [(True, "integer"), (1, "2 <="), (4, "floor")],
)
def test_cross_validation_rejects_invalid_automatic_splits_before_fitting(
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
    n_splits,
    error,
):
    """Automatic split counts are validated before any model is constructed."""
    from phylognn import cross_validate_leaf_regression, prepare_leaf_regression

    data = prepare_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
    )
    _InvalidOutputRegressor.instances.clear()
    with pytest.raises((TypeError, ValueError), match=error):
        cross_validate_leaf_regression(
            data,
            n_splits=n_splits,
            model_class=_InvalidOutputRegressor,
            model_config={"kind": "shape"},
        )
    assert not _InvalidOutputRegressor.instances


def test_cross_validation_rejects_constant_default_r2_targets_before_fitting(
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_tree,
):
    """Default R-squared folds require nonconstant validation targets before fitting."""
    from phylognn import cross_validate_leaf_regression, prepare_leaf_regression

    data = prepare_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        torch.ones(6, dtype=torch.float32),
    )
    _InvalidOutputRegressor.instances.clear()
    with pytest.raises(ValueError, match="R-squared"):
        cross_validate_leaf_regression(
            data,
            validation_folds=([0, 1], [2, 3], [4, 5]),
            model_class=_InvalidOutputRegressor,
            model_config={"kind": "shape"},
        )
    assert not _InvalidOutputRegressor.instances


@pytest.mark.parametrize("entry_point", ["cross_validate", "workflow"])
def test_invalid_workflow_contracts_restore_caller_rng_state(
    entry_point,
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
):
    """CV and the facade restore Python, NumPy, and PyTorch RNG states on errors."""
    from phylognn import (
        LeafRegressionConfig,
        cross_validate_leaf_regression,
        prepare_leaf_regression,
        run_leaf_regression,
    )

    random.seed(73)
    np.random.seed(73)
    torch.manual_seed(73)
    expected = (random.random(), np.random.rand(), torch.rand(2))
    random.seed(73)
    np.random.seed(73)
    torch.manual_seed(73)

    def invalid_score(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        del predictions, targets
        random.random()
        np.random.rand()
        torch.rand(1)
        return float("nan")

    kwargs = {
        "validation_folds": ([0, 1], [2, 3], [4, 5]),
        "training_config": LeafRegressionConfig(epochs=1, seed=31),
        "score_fn": invalid_score,
    }
    with pytest.raises(ValueError, match="score_fn"):
        if entry_point == "cross_validate":
            data = prepare_leaf_regression(
                leaf_regression_tree,
                leaf_regression_representations,
                leaf_regression_position_mask,
                leaf_regression_targets,
            )
            cross_validate_leaf_regression(data, **kwargs)
        else:
            run_leaf_regression(
                leaf_regression_tree,
                leaf_regression_representations,
                leaf_regression_position_mask,
                leaf_regression_targets,
                **kwargs,
            )

    actual = (random.random(), np.random.rand(), torch.rand(2))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])
