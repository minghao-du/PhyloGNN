"""Tests for the recommended leaf-regression workflow."""

import copy
import inspect
import random
import warnings

import numpy as np
import pytest
import torch


def test_fit_leaf_regression_public_exports_and_pgls_signature():
    import phylognn
    import phylognn.leaf_regression as leaf_regression

    fit = leaf_regression.fit_leaf_regression
    assert phylognn.fit_leaf_regression is fit
    assert "fit_leaf_regression" in leaf_regression.__all__
    assert "fit_leaf_regression" in phylognn.__all__

    signature = inspect.signature(fit)
    for name in ("pgls_head", "pgls_loss", "covariances", "batch"):
        parameter = signature.parameters[name]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is None


def test_prepare_leaf_regression_defers_representation_finiteness_to_model(
    leaf_regression_tree,
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
):
    """Data construction validates structure but leaves raw finiteness to models."""
    from phylognn.leaf_regression import prepare_leaf_regression

    representations = leaf_regression_representations.clone()
    representations[0, 0, 0] = float("nan")

    data = prepare_leaf_regression(
        leaf_regression_tree,
        representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
    )

    assert torch.isnan(data.representations[0, 0, 0])


@pytest.mark.parametrize("mask", [torch.ones(6, 4), torch.tensor([[1, 0, 0, 0]] * 6)])
def test_prepare_leaf_regression_canonicalizes_numeric_binary_masks(
    mask,
    leaf_regression_tree,
    leaf_regression_representations,
    leaf_regression_targets,
):
    """Accepted numeric zero/one masks become boolean data fields."""
    from phylognn.leaf_regression import prepare_leaf_regression

    data = prepare_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        mask,
        leaf_regression_targets,
    )

    assert data.position_mask.dtype == torch.bool


@pytest.mark.parametrize(
    "position_mask",
    [torch.ones(6, 4, dtype=torch.float32), torch.tensor([[1, 0, 0, 0]] * 6)],
)
def test_leaf_regression_data_construction_canonicalizes_numeric_binary_masks(
    position_mask,
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
):
    """Direct data construction stores accepted numeric masks as bool tensors."""
    from phylognn.leaf_regression import LeafRegressionData, prepare_leaf_regression

    prepared = prepare_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
    )

    data = LeafRegressionData(
        leaf_names=prepared.leaf_names,
        representations=prepared.representations,
        position_mask=position_mask,
        targets=prepared.targets,
        leaf_laplacian=prepared.leaf_laplacian,
    )

    assert data.position_mask.dtype == torch.bool
    assert torch.equal(data.position_mask, position_mask.to(dtype=torch.bool))


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("representations", torch.ones(6, 4), "representations"),
        ("position_mask", torch.ones(6, 3, dtype=torch.bool), "position_mask"),
        ("position_mask", torch.full((6, 4), 0.5), "position_mask"),
    ],
)
def test_leaf_regression_data_validates_structure_and_mask_contracts(
    field,
    value,
    match,
    leaf_regression_tree,
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
):
    """Malformed structural fields fail at the data boundary."""
    from phylognn.leaf_regression import prepare_leaf_regression

    kwargs = {
        "representations": leaf_regression_representations,
        "position_mask": leaf_regression_position_mask,
        "targets": leaf_regression_targets,
    }
    kwargs[field] = value
    with pytest.raises((TypeError, ValueError), match=match):
        prepare_leaf_regression(leaf_regression_tree, **kwargs)


def test_prepare_leaf_regression_rejects_non_contiguous_right_padding(
    leaf_regression_tree,
    leaf_regression_representations,
    leaf_regression_targets,
):
    """Preparation rejects internal mask gaps before any model can be constructed."""
    from phylognn.leaf_regression import prepare_leaf_regression

    mask = torch.tensor([[1, 0, 1, 0]] * 6, dtype=torch.bool)
    with pytest.raises(ValueError, match="contiguous right padding"):
        prepare_leaf_regression(
            leaf_regression_tree,
            leaf_regression_representations,
            mask,
            leaf_regression_targets,
        )


def test_leaf_regression_data_rejects_non_contiguous_right_padding(
    leaf_regression_tree,
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
):
    """Direct data construction applies the same right-padding contract."""
    from phylognn.leaf_regression import LeafRegressionData, prepare_leaf_regression

    prepared = prepare_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
    )
    mask = torch.tensor([[1, 0, 1, 0]] * 6, dtype=torch.bool)

    with pytest.raises(ValueError, match="contiguous right padding"):
        LeafRegressionData(
            leaf_names=prepared.leaf_names,
            representations=prepared.representations,
            position_mask=mask,
            targets=prepared.targets,
            leaf_laplacian=prepared.leaf_laplacian,
        )


class _ScriptedLossRegressor(torch.nn.Module):
    """Return stateful scripted outputs for best-state restoration tests."""

    instances: list["_ScriptedLossRegressor"] = []

    def __init__(self, prediction_script: tuple[tuple[float, ...], ...]) -> None:
        super().__init__()
        self.register_buffer(
            "prediction_script", torch.tensor(prediction_script, dtype=torch.float32)
        )
        self.register_buffer("script_index", torch.zeros((), dtype=torch.long))
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        type(self).instances.append(self)

    def forward(
        self, representations: torch.Tensor, position_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        script_index = min(self.script_index.item(), self.prediction_script.size(0) - 1)
        predictions = self.prediction_script[script_index].to(representations) + self.weight
        position_weights = torch.arange(
            1, position_mask.size(1) + 1, dtype=representations.dtype, device=representations.device
        )
        attention = position_mask.to(dtype=representations.dtype) * (
            position_weights * (1 + torch.sigmoid(self.weight)) + script_index
        )
        attention = attention / attention.sum(dim=1, keepdim=True)
        self.script_index.add_(1)
        return predictions, attention


class _RefitDurationRegressor(torch.nn.Module):
    """Use construction-ordered prediction scripts for fold/refit duration tests."""

    instances: list["_RefitDurationRegressor"] = []
    configured_stage_scripts: tuple[tuple[tuple[float, ...], ...], ...] = ()

    def __init__(
        self, stage_scripts: tuple[tuple[tuple[float, ...], ...], ...] | None = None
    ) -> None:
        super().__init__()
        stage_scripts = stage_scripts or type(self).configured_stage_scripts
        stage_index = len(type(self).instances)
        if stage_index >= len(stage_scripts):
            raise AssertionError("The test did not provide a script for this model stage.")
        self.prediction_script = torch.tensor(stage_scripts[stage_index], dtype=torch.float32)
        self.script_index = torch.zeros((), dtype=torch.long)
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        type(self).instances.append(self)

    def forward(self, representations: torch.Tensor, position_mask: torch.Tensor) -> torch.Tensor:
        del position_mask
        script_index = min(self.script_index.item(), self.prediction_script.size(0) - 1)
        self.script_index.add_(1)
        return self.prediction_script[script_index].to(representations) + self.weight * 0


class _ConstructionSentinelRegressor(torch.nn.Module):
    """Record construction so preflight failures can prove no model was created."""

    construction_count = 0

    def __init__(self) -> None:
        super().__init__()
        type(self).construction_count += 1
        self.weight = torch.nn.Parameter(torch.tensor(0.1))

    def forward(self, representations: torch.Tensor, position_mask: torch.Tensor) -> torch.Tensor:
        del position_mask
        return representations[:, 0, 0] * self.weight


def test_scripted_loss_regressor_restores_predictions_and_attention_from_state_dict():
    """The scripted double supports complete best-state restoration assertions."""
    model = _ScriptedLossRegressor(((0.0, 1.0), (2.0, 3.0)))
    representations = torch.zeros((2, 3, 1), dtype=torch.float32)
    position_mask = torch.tensor([[True, True, True], [True, True, False]])

    saved_state = copy.deepcopy(model.state_dict())
    first_predictions, first_attention = model(representations, position_mask)
    second_predictions, second_attention = model(representations, position_mask)
    model.load_state_dict(saved_state)
    restored_predictions, restored_attention = model(representations, position_mask)

    torch.testing.assert_close(restored_predictions, first_predictions)
    torch.testing.assert_close(restored_attention, first_attention)
    assert not torch.equal(first_predictions, second_predictions)
    assert not torch.equal(first_attention, second_attention)


# Fixed outputs captured before leaf-regression early stopping was introduced.
_FIXED_EPOCH_BASELINE = {
    "config": {"epochs": 3, "learning_rate": 0.01, "seed": 29, "offset": 0.25},
    "fit": {
        "losses": (1.223833441734314, 1.2234065532684326, 1.2230627536773682),
        "predictions": (
            0.37962841987609863,
            0.25,
            0.28888851404190063,
            0.3666655719280243,
            0.340739905834198,
            0.26296284794807434,
        ),
    },
    "manual_cv": {
        "fold_losses": (
            (1.2744998931884766, 1.2670848369598389, 1.2597417831420898),
            (1.7557499408721924, 1.7540873289108276, 1.7525019645690918),
            (0.6412500143051147, 0.636947512626648, 0.6327426433563232),
        ),
        "fold_scores": (1.163437008857727, 0.17477791011333466, 2.4195008277893066),
        "cv_score": 1.2525719155867894,
        "oof_predictions": (
            0.37999051809310913,
            0.25,
            0.27101603150367737,
            0.3130480647087097,
            0.29901647567749023,
            0.2570023536682129,
        ),
    },
    "automatic_cv": {
        "validation_folds": ((5, 4), (2, 1), (3, 0)),
        "fold_losses": (
            (0.6412500143051147, 0.636947512626648, 0.6327426433563232),
            (1.728524923324585, 1.7277626991271973, 1.7271199226379395),
            (1.301724910736084, 1.2968096733093262, 1.2919245958328247),
        ),
        "fold_scores": (2.4195008277893066, 0.21520331501960754, 1.0942392349243164),
        "cv_score": 1.2429811259110768,
        "oof_predictions": (
            0.37999409437179565,
            0.25,
            0.2889195680618286,
            0.3669946789741516,
            0.29901647567749023,
            0.2570023536682129,
        ),
    },
    "workflow": {
        "cv_score": 1.2525719155867894,
        "predictions": (
            0.37962841987609863,
            0.25,
            0.28888851404190063,
            0.3666655719280243,
            0.340739905834198,
            0.26296284794807434,
        ),
    },
}


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


class _LeafRepresentationBackbone(torch.nn.Module):
    """Minimal custom provider of ordered leaf representations for PGLS tests."""

    instances: list["_LeafRepresentationBackbone"] = []

    def __init__(self, input_dim: int = 3, output_dim: int = 2) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(input_dim, output_dim)
        self.received_representations: torch.Tensor | None = None
        self.received_position_mask: torch.Tensor | None = None
        type(self).instances.append(self)

    def forward_leaf_representations(
        self, representations: torch.Tensor, position_mask: torch.Tensor
    ) -> torch.Tensor:
        """Pool validated ``[N, L, D_input]`` inputs into ordered ``[N, D]`` features."""
        self.received_representations = representations
        self.received_position_mask = position_mask
        mask = position_mask.unsqueeze(-1).to(dtype=representations.dtype)
        pooled = (representations * mask).sum(dim=1) / mask.sum(dim=1)
        return self.projection(pooled)


class _NonCallableLeafRepresentationBackbone(_LeafRepresentationBackbone):
    forward_leaf_representations = None


class _MissingLeafRepresentationBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))


class _IncompatibleLeafRepresentationBackbone(_LeafRepresentationBackbone):
    def forward_leaf_representations(self, representations: torch.Tensor) -> torch.Tensor:
        return representations[:, 0]


def _pgls_fitting_covariances(
    *, dtype: torch.dtype = torch.float32, device: torch.device | str = "cpu"
) -> list[torch.Tensor]:
    """Return known SPD covariance blocks for three ordered two-leaf trees."""
    return [
        torch.tensor([[1.0, 0.2], [0.2, 0.9]], dtype=dtype, device=device),
        torch.tensor([[1.1, 0.1], [0.1, 0.8]], dtype=dtype, device=device),
        torch.tensor([[0.7, 0.15], [0.15, 1.2]], dtype=dtype, device=device),
    ]


def _pgls_fitting_batch(*, device: torch.device | str = "cpu") -> torch.Tensor:
    """Map six ordered leaves to the three covariance blocks used by fitting tests."""
    return torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long, device=device)


class _RecordingMultiTraitPGLSLoss(torch.nn.Module):
    """Record fitting subsets while applying PGLS to repeated scalar targets."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[list[torch.Tensor], torch.Tensor]] = []

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        covariances: list[torch.Tensor],
        batch: torch.Tensor,
    ) -> torch.Tensor:
        from phylognn.training import PGLSLoss

        self.calls.append(
            ([covariance.detach().clone() for covariance in covariances], batch.detach().clone())
        )
        expanded_targets = targets.unsqueeze(1).expand_as(predictions)
        return PGLSLoss()(predictions, expanded_targets, covariances, batch)


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

    @property
    def ordered_payloads(self):
        """Return leaf metric payloads in tracker order."""
        return [payload for _, payload in self.metric_calls]

    @property
    def fold_payloads(self):
        """Return ordered fold-level payloads."""
        return [payload for payload in self.ordered_payloads if "cv/fold_score" in payload]

    @property
    def summary_payloads(self):
        """Return ordered CV summary payloads."""
        return [payload for payload in self.ordered_payloads if "cv/mean_score" in payload]

    @property
    def stage_payloads(self):
        """Return payloads carrying stage identity, preserving event order."""
        return [payload for payload in self.ordered_payloads if "stage/type" in payload]

    @property
    def terminal_payloads(self):
        """Return terminal status payloads in the order they were logged."""
        return [payload for payload in self.ordered_payloads if "status/state" in payload]

    def assert_stage_epochs(
        self, stage_type: str, stage_index: int, expected_epochs: list[int]
    ) -> None:
        """Assert the ordered epoch numbers emitted for one tracked stage."""
        actual_epochs = [
            payload["stage/epoch"]
            for payload in self.ordered_payloads
            if payload.get("stage/type") == stage_type
            and payload.get("stage/index") == stage_index
            and "stage/epoch" in payload
        ]
        assert actual_epochs == expected_epochs


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
    coordinator.log_epoch(
        fit_stage,
        train_predictions=torch.tensor([1.0, 3.0]),
        train_targets=torch.tensor([2.0, 4.0]),
        loss_fn=torch.nn.functional.mse_loss,
        learning_rate=0.01,
        epoch_time_sec=0.2,
    )
    cv_stage = coordinator.start_stage("cv_fold")
    coordinator.log_epoch(
        cv_stage,
        train_predictions=torch.tensor([1.0, 3.0]),
        train_targets=torch.tensor([2.0, 4.0]),
        val_predictions=torch.tensor([1.0, 3.0]),
        val_targets=torch.tensor([2.0, 4.0]),
        loss_fn=torch.nn.functional.mse_loss,
        learning_rate=0.01,
        epoch_time_sec=0.3,
    )
    coordinator.finish("completed")
    coordinator.finish("failed")

    assert tracker.start_payloads == [{"data.config_file": "input.toml", "workflow.type": "fit"}]
    assert [step for step, _ in tracker.metric_calls] == [1, 2, 2]
    assert tracker.metric_calls[0][1] == {
        "stage/epoch": 1,
        "stage/index": 1,
        "stage/type": "fit",
        "train/loss": 1.0,
        "train/score": 0.0,
        "train/mae": 1.0,
        "train/pearson_r": 1.0,
        "train/lr": 0.01,
        "train/epoch_time_sec": 0.2,
    }
    assert tracker.metric_calls[1][1]["stage/type"] == "cv_fold"
    assert {
        "train/loss",
        "train/score",
        "train/mae",
        "train/pearson_r",
        "val/loss",
        "val/score",
        "val/mae",
        "val/pearson_r",
    } <= tracker.metric_calls[1][1].keys()
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
    disabled.log_epoch(
        disabled.start_stage("fit"),
        train_predictions=torch.tensor([1.0]),
        train_targets=torch.tensor([1.0]),
        loss_fn=torch.nn.functional.mse_loss,
        learning_rate=0.1,
        epoch_time_sec=0.0,
    )
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


def test_leaf_tracking_epoch_metrics_warn_once_for_undefined_pearson_and_reject_invalid_pairs():
    """Epoch metrics use complete aligned partitions and validate before logging."""
    from phylognn.leaf_regression.tracking import _LeafExperimentCoordinator
    from phylognn.training.tracking import TrackingConfig, TrackingError

    tracker = _ObservableLeafTracker()
    coordinator = _LeafExperimentCoordinator(
        TrackingConfig(enabled=True, project="phylognn"), tracker=tracker
    )
    coordinator.start({"workflow.type": "fit"})
    stage = coordinator.start_stage("fit")
    with pytest.warns(RuntimeWarning, match="Pearson"):
        coordinator.log_epoch(
            stage,
            train_predictions=torch.tensor([2.0]),
            train_targets=torch.tensor([1.0]),
            score_fn=lambda *_: 0.0,
            loss_fn=torch.nn.functional.mse_loss,
            learning_rate=0.1,
            epoch_time_sec=0.0,
        )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        coordinator.log_epoch(
            stage,
            train_predictions=torch.tensor([3.0]),
            train_targets=torch.tensor([1.0]),
            score_fn=lambda *_: 0.0,
            loss_fn=torch.nn.functional.mse_loss,
            learning_rate=0.1,
            epoch_time_sec=0.0,
        )
    assert not caught
    assert all("train/pearson_r" not in payload for _, payload in tracker.metric_calls)

    for predictions, targets, score_fn, pattern in (
        (torch.tensor([]), torch.tensor([]), None, "nonempty"),
        (torch.tensor([1.0]), torch.tensor([1.0, 2.0]), None, "aligned"),
        (torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0]), lambda *_: float("nan"), "score"),
    ):
        with pytest.raises(TrackingError, match=pattern):
            coordinator.log_epoch(
                stage,
                train_predictions=predictions,
                train_targets=targets,
                score_fn=score_fn,
                loss_fn=torch.nn.functional.mse_loss,
                learning_rate=0.1,
                epoch_time_sec=0.0,
            )
    assert len(tracker.metric_calls) == 2


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


def test_stage_config_forwards_loss_selection_to_every_fold_and_refit(
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
):
    """Every CV fold and the refit train with the selected loss, not a silent MSE fallback."""
    from phylognn import (
        LeafRegressionConfig,
        cross_validate_leaf_regression,
        prepare_leaf_regression,
    )
    from phylognn.leaf_regression.validation import _stage_config

    data = prepare_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
    )
    folds = ([0, 1], [2, 3], [4, 5])
    config = LeafRegressionConfig(epochs=1, learning_rate=0.01, seed=19, loss="mae")
    result = cross_validate_leaf_regression(
        data,
        validation_folds=folds,
        training_config=config,
        score_fn=_fold_size_score,
        refit=True,
        model_class=_ConfiguredRegressor,
        model_config={"offset": 0.2},
    )

    all_indices = torch.arange(len(data.leaf_names), dtype=torch.long)
    for fold, fold_result in zip(folds, result.fold_results, strict=True):
        train_indices = all_indices[~torch.isin(all_indices, torch.tensor(fold))]
        expected_predictions = leaf_regression_representations[train_indices, 0, 0] * 0.1 + 0.2
        expected_loss = torch.nn.functional.l1_loss(
            expected_predictions, leaf_regression_targets[train_indices]
        ).item()
        assert fold_result.losses[0] == pytest.approx(expected_loss)

    assert result.final_fit is not None
    expected_refit_predictions = leaf_regression_representations[:, 0, 0] * 0.1 + 0.2
    expected_refit_loss = torch.nn.functional.l1_loss(
        expected_refit_predictions, leaf_regression_targets
    ).item()
    assert result.final_fit.losses[0] == pytest.approx(expected_refit_loss)

    unseeded_config = LeafRegressionConfig(loss="huber", huber_delta=2.0)
    assert _stage_config(unseeded_config, 3) is unseeded_config


def test_stage_config_forwards_huber_delta_to_every_fold_and_refit(
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
):
    """Every CV fold and the refit train with the configured delta, not the `1.0` default."""
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
    folds = ([0, 1], [2, 3], [4, 5])
    config = LeafRegressionConfig(
        epochs=1, learning_rate=0.01, seed=19, loss="huber", huber_delta=0.5
    )
    result = cross_validate_leaf_regression(
        data,
        validation_folds=folds,
        training_config=config,
        score_fn=_fold_size_score,
        refit=True,
        model_class=_ConfiguredRegressor,
        model_config={"offset": 0.2},
    )

    all_indices = torch.arange(len(data.leaf_names), dtype=torch.long)
    for fold, fold_result in zip(folds, result.fold_results, strict=True):
        train_indices = all_indices[~torch.isin(all_indices, torch.tensor(fold))]
        expected_predictions = leaf_regression_representations[train_indices, 0, 0] * 0.1 + 0.2
        expected_loss = torch.nn.HuberLoss(delta=0.5)(
            expected_predictions, leaf_regression_targets[train_indices]
        ).item()
        default_delta_loss = torch.nn.HuberLoss(delta=1.0)(
            expected_predictions, leaf_regression_targets[train_indices]
        ).item()
        assert fold_result.losses[0] == pytest.approx(expected_loss)
        assert fold_result.losses[0] != pytest.approx(default_delta_loss)

    assert result.final_fit is not None
    expected_refit_predictions = leaf_regression_representations[:, 0, 0] * 0.1 + 0.2
    expected_refit_loss = torch.nn.HuberLoss(delta=0.5)(
        expected_refit_predictions, leaf_regression_targets
    ).item()
    assert result.final_fit.losses[0] == pytest.approx(expected_refit_loss)


def test_cross_validate_leaf_regression_is_deterministic_with_huber_delta(
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
):
    """Two identical Huber-delta CV runs reproduce folds, predictions, and loss histories."""
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
    config = LeafRegressionConfig(
        epochs=2, learning_rate=0.01, seed=23, loss="huber", huber_delta=0.5
    )

    def _run():
        return cross_validate_leaf_regression(
            data,
            n_splits=3,
            training_config=config,
            refit=True,
            model_class=_ConfiguredRegressor,
            model_config={"offset": 0.2},
        )

    first_result = _run()
    second_result = _run()

    assert first_result.oof_predictions.tolist() == second_result.oof_predictions.tolist()
    for first_fold, second_fold in zip(
        first_result.fold_results, second_result.fold_results, strict=True
    ):
        assert torch.equal(first_fold.train_indices, second_fold.train_indices)
        torch.testing.assert_close(first_fold.predictions, second_fold.predictions)
        assert first_fold.losses == second_fold.losses
    assert first_result.final_fit is not None
    assert second_result.final_fit is not None
    torch.testing.assert_close(
        first_result.final_fit.predictions, second_result.final_fit.predictions
    )
    assert first_result.final_fit.losses == second_result.final_fit.losses


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
    summaries = tracker.summary_payloads
    assert len(summaries) == 1
    assert {
        "cv/mean_score",
        "cv/weighted_score",
        "cv/std_score",
        "cv/min_score",
        "cv/max_score",
        "cv/mae",
        "cv/pearson_r",
    } <= summaries[0].keys()
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


@pytest.mark.parametrize(
    "training_config_kwargs, expected_identifier",
    [
        ({}, "mse"),
        ({"loss": "mae"}, "mae"),
        ({"loss": "huber"}, "huber(delta=1.0)"),
    ],
)
def test_loss_identifier_metadata_is_consistent_across_all_start_payload_entry_points(
    training_config_kwargs,
    expected_identifier,
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
):
    """``loss.name`` carries the resolved identifier for fit, CV, and the full workflow."""
    from phylognn import (
        LeafRegressionConfig,
        cross_validate_leaf_regression,
        fit_leaf_regression,
        prepare_leaf_regression,
        run_leaf_regression,
    )
    from phylognn.training.tracking import TrackingConfig

    data = prepare_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
    )
    tracking_config = TrackingConfig(enabled=True, project="phylognn")
    config = LeafRegressionConfig(epochs=1, learning_rate=0.01, seed=7, **training_config_kwargs)

    fit_tracker = _ObservableLeafTracker()
    fit_leaf_regression(
        data,
        train_indices=[0, 2, 4],
        training_config=config,
        model_class=_ConfiguredRegressor,
        model_config={"offset": 0.2},
        tracking_config=tracking_config,
        tracker=fit_tracker,
    )
    assert fit_tracker.start_payloads[0]["loss.name"] == expected_identifier
    fit_epoch_metrics = next(
        metrics for _, metrics in fit_tracker.metric_calls if "train/loss" in metrics
    )
    assert {"train/loss", "train/score", "train/mae"} <= fit_epoch_metrics.keys()

    cv_tracker = _ObservableLeafTracker()
    cross_validate_leaf_regression(
        data,
        validation_folds=([0, 1], [2, 3], [4, 5]),
        training_config=config,
        score_fn=_fold_size_score,
        refit=False,
        model_class=_ConfiguredRegressor,
        model_config={"offset": 0.2},
        tracking_config=tracking_config,
        tracker=cv_tracker,
    )
    assert cv_tracker.start_payloads[0]["loss.name"] == expected_identifier
    cv_epoch_metrics = next(
        metrics for _, metrics in cv_tracker.metric_calls if "val/loss" in metrics
    )
    assert {"val/loss", "val/score", "val/mae"} <= cv_epoch_metrics.keys()

    run_tracker = _ObservableLeafTracker()
    run_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
        validation_folds=([0, 1], [2, 3], [4, 5]),
        training_config=config,
        score_fn=_fold_size_score,
        model_class=_ConfiguredRegressor,
        model_config={"offset": 0.2},
        tracking_config=tracking_config,
        tracker=run_tracker,
    )
    assert run_tracker.start_payloads[0]["loss.name"] == expected_identifier


@pytest.mark.parametrize(
    "huber_delta, expected_identifier",
    [
        (1, "huber(delta=1.0)"),
        (1.0, "huber(delta=1.0)"),
        (1.5, "huber(delta=1.5)"),
    ],
)
def test_huber_delta_identifier_metadata_groups_and_distinguishes_across_entry_points(
    huber_delta,
    expected_identifier,
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
):
    """`loss.name` groups numerically equal deltas and distinguishes differing ones."""
    from phylognn import (
        LeafRegressionConfig,
        cross_validate_leaf_regression,
        fit_leaf_regression,
        prepare_leaf_regression,
        run_leaf_regression,
    )
    from phylognn.training.tracking import TrackingConfig

    data = prepare_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
    )
    tracking_config = TrackingConfig(enabled=True, project="phylognn")
    config = LeafRegressionConfig(
        epochs=1, learning_rate=0.01, seed=7, loss="huber", huber_delta=huber_delta
    )
    metadata_field_names = None

    fit_tracker = _ObservableLeafTracker()
    fit_leaf_regression(
        data,
        train_indices=[0, 2, 4],
        training_config=config,
        model_class=_ConfiguredRegressor,
        model_config={"offset": 0.2},
        tracking_config=tracking_config,
        tracker=fit_tracker,
    )
    fit_payload = fit_tracker.start_payloads[0]
    assert fit_payload["loss.name"] == expected_identifier
    metadata_field_names = set(fit_payload)

    cv_tracker = _ObservableLeafTracker()
    cross_validate_leaf_regression(
        data,
        validation_folds=([0, 1], [2, 3], [4, 5]),
        training_config=config,
        score_fn=_fold_size_score,
        refit=False,
        model_class=_ConfiguredRegressor,
        model_config={"offset": 0.2},
        tracking_config=tracking_config,
        tracker=cv_tracker,
    )
    cv_payload = cv_tracker.start_payloads[0]
    assert cv_payload["loss.name"] == expected_identifier
    assert set(cv_payload) == metadata_field_names

    run_tracker = _ObservableLeafTracker()
    run_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
        validation_folds=([0, 1], [2, 3], [4, 5]),
        training_config=config,
        score_fn=_fold_size_score,
        model_class=_ConfiguredRegressor,
        model_config={"offset": 0.2},
        tracking_config=tracking_config,
        tracker=run_tracker,
    )
    run_payload = run_tracker.start_payloads[0]
    assert run_payload["loss.name"] == expected_identifier
    assert set(run_payload) == metadata_field_names


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
        assert {"train/loss", "train/score", "train/mae"} <= metrics.keys()
        assert not any(key.startswith("val/") for key in metrics)
        assert all(np.isfinite(metrics[key]) for key in ("train/loss", "train/score", "train/mae"))
        assert metrics["train/lr"] == 0.01
        assert np.isfinite(metrics["train/epoch_time_sec"])
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
    for _, metrics in epoch_calls:
        assert {
            "train/loss",
            "train/score",
            "train/mae",
            "train/pearson_r",
            "val/loss",
            "val/score",
            "val/mae",
            "val/pearson_r",
        } <= metrics.keys()
        assert all(
            np.isfinite(metrics[key]) for key in metrics if key.startswith(("train/", "val/"))
        )
    summaries = [metrics for _, metrics in tracker.metric_calls if "cv/fold_score" in metrics]
    assert [summary["stage/index"] for summary in summaries] == [1, 2, 3]
    assert [summary["cv/validation_leaf_count"] for summary in summaries] == [2, 2, 2]
    assert [summary["cv/fold_score"] for summary in summaries] == [2.0, 2.0, 2.0]
    assert len(tracker.summary_payloads) == 1
    assert tracker.summary_payloads[0]["cv/mean_score"] == 2.0
    assert tracker.summary_payloads[0]["cv/weighted_score"] == 2.0
    assert tracker.summary_payloads[0]["cv/std_score"] == 0.0
    assert tracker.summary_payloads[0]["cv/min_score"] == 2.0
    assert tracker.summary_payloads[0]["cv/max_score"] == 2.0
    assert np.isfinite(tracker.summary_payloads[0]["cv/mae"])
    assert np.isfinite(tracker.summary_payloads[0]["cv/pearson_r"])
    assert result.cv_score == pytest.approx(2.0)
    assert result.final_fit is None
    assert not any(metrics.get("stage/type") == "refit" for _, metrics in tracker.metric_calls)
    assert tracker.metric_calls[-1] == (6, {"status/state": "completed"})
    assert tracker.finish_calls == ["completed"]


def test_leaf_cv_summary_omits_undefined_pearson_and_rejects_nonfinite_payloads():
    """Undefined correlations warn without suppressing other finite summaries."""
    from phylognn.leaf_regression.tracking import _LeafExperimentCoordinator
    from phylognn.leaf_regression.validation import _build_cv_summary_metrics
    from phylognn.training.tracking import TrackingConfig, TrackingError

    with pytest.warns(RuntimeWarning, match="Pearson"):
        summary = _build_cv_summary_metrics(
            scores=(1.0, 3.0),
            folds=(torch.tensor([0]), torch.tensor([1])),
            oof_predictions=torch.tensor([2.0, 2.0]),
            targets=torch.tensor([1.0, 2.0]),
        )

    assert summary == {
        "cv/mean_score": 2.0,
        "cv/weighted_score": 2.0,
        "cv/std_score": 1.0,
        "cv/min_score": 1.0,
        "cv/max_score": 3.0,
        "cv/mae": 0.5,
    }

    coordinator = _LeafExperimentCoordinator(TrackingConfig(enabled=True, project="phylognn"))
    coordinator._tracker = _ObservableLeafTracker()
    coordinator._started = True
    with pytest.raises(TrackingError, match="finite"):
        coordinator.log_summary({"cv/mean_score": float("nan")})


@pytest.mark.parametrize(
    "selection, expected_epoch_keys, expected_fold_keys, expected_summary_keys",
    [
        (
            ("train/loss", "cv/fold_score", "cv/mae"),
            {"stage/type", "stage/index", "stage/epoch", "train/loss"},
            {"stage/type", "stage/index", "cv/fold_score"},
            {"cv/mae"},
        ),
        ((), {"stage/type", "stage/index", "stage/epoch"}, {"stage/type", "stage/index"}, set()),
        (
            ("val/loss",),
            {"stage/type", "stage/index", "stage/epoch", "val/loss"},
            {"stage/type", "stage/index"},
            set(),
        ),
    ],
)
def test_leaf_tracking_metric_selection_filters_payloads_and_keeps_operational_fields(
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
    selection,
    expected_epoch_keys,
    expected_fold_keys,
    expected_summary_keys,
):
    """Leaf CV filters quantitative values without changing stages or results."""
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
        training_config=LeafRegressionConfig(epochs=1, learning_rate=0.01, seed=53),
        score_fn=_fold_size_score,
        refit=False,
        model_class=_ConfiguredRegressor,
        model_config={"offset": 0.2},
        tracking_config=TrackingConfig(enabled=True, project="phylognn", metrics=selection),
        tracker=tracker,
    )

    fold_payloads = [
        payload
        for payload in tracker.stage_payloads
        if payload["stage/type"] == "cv_fold" and "stage/epoch" not in payload
    ]
    summary_payloads = [
        payload
        for payload in tracker.ordered_payloads
        if "cv/mae" in payload or "cv/mean_score" in payload
    ]
    assert set(fold_payloads[0]) == expected_fold_keys
    assert (set(summary_payloads[0]) if summary_payloads else set()) == expected_summary_keys
    epoch_payload = next(payload for payload in tracker.stage_payloads if "stage/epoch" in payload)
    assert set(epoch_payload) == expected_epoch_keys
    assert tracker.terminal_payloads == [{"status/state": "completed"}]
    assert result.cv_score == pytest.approx(2.0)


@pytest.mark.parametrize(
    "selection, pattern",
    [
        (("train/loss", "train/loss"), "duplicate"),
        (("loss",), "namespace/name"),
        (("train/api-key",), "sensitive"),
        (("cv/not_a_metric",), "unknown"),
        (("train/not_configured",), "unknown"),
        (("train/loss", 1), "only strings"),
    ],
)
def test_leaf_tracking_rejects_invalid_selection_before_tracker_start(selection, pattern):
    """Leaf tracking rejects structural and workflow-invalid selections before start."""
    from phylognn.leaf_regression.tracking import _LeafExperimentCoordinator
    from phylognn.training.tracking import TrackingConfig, TrackingError

    tracker = _ObservableLeafTracker()
    with pytest.raises(TrackingError, match=pattern):
        _LeafExperimentCoordinator(
            TrackingConfig(enabled=True, project="phylognn", metrics=selection), tracker=tracker
        )
    assert tracker.start_payloads == []


def test_leaf_tracking_selection_is_deterministic_and_preserves_cv_results(
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
):
    """Equivalent selected runs retain their scientific result and payload keys."""
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
    kwargs = {
        "validation_folds": ([0, 1], [2, 3], [4, 5]),
        "training_config": LeafRegressionConfig(epochs=1, learning_rate=0.01, seed=59),
        "score_fn": _fold_size_score,
        "refit": False,
        "model_class": _ConfiguredRegressor,
        "model_config": {"offset": 0.2},
    }
    untracked = cross_validate_leaf_regression(data, **kwargs)
    tracked_payload_keys = []
    for _ in range(2):
        tracker = _ObservableLeafTracker()
        tracked = cross_validate_leaf_regression(
            data,
            tracking_config=TrackingConfig(
                enabled=True,
                project="phylognn",
                metrics=("train/loss", "cv/fold_score", "cv/mae"),
            ),
            tracker=tracker,
            **kwargs,
        )
        assert tracked.cv_score == untracked.cv_score
        assert tracked.fold_scores == untracked.fold_scores
        assert torch.equal(tracked.oof_predictions, untracked.oof_predictions)
        tracked_payload_keys.append([tuple(payload) for payload in tracker.ordered_payloads])

    assert tracked_payload_keys[0] == tracked_payload_keys[1]


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
    monkeypatch,
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

    captured_optimizers = []
    original_adam = torch.optim.Adam

    def capture_adam(*args, **kwargs):
        optimizer = original_adam(*args, **kwargs)
        captured_optimizers.append(optimizer)
        return optimizer

    monkeypatch.setattr("phylognn.leaf_regression.fitting.torch.optim.Adam", capture_adam)

    fit_without_tracking = fit_leaf_regression(data, training_config=config, **model_kwargs)
    fit_with_tracking = fit_leaf_regression(
        data,
        training_config=config,
        tracking_config=tracked_config,
        tracker=_ObservableLeafTracker(),
        **model_kwargs,
    )
    assert len(captured_optimizers) == 2
    untracked_adam_state, tracked_adam_state = (
        copy.deepcopy(optimizer.state_dict()) for optimizer in captured_optimizers
    )
    assert set(untracked_adam_state["state"]) == set(tracked_adam_state["state"])
    for parameter_id, untracked_state in untracked_adam_state["state"].items():
        tracked_state = tracked_adam_state["state"][parameter_id]
        assert set(untracked_state) == set(tracked_state)
        for field_name, untracked_value in untracked_state.items():
            tracked_value = tracked_state[field_name]
            assert torch.is_tensor(untracked_value), field_name
            torch.testing.assert_close(untracked_value, tracked_value, rtol=1e-6, atol=1e-7)

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
    torch.testing.assert_close(
        fit_with_tracking.predictions, fit_without_tracking.predictions, rtol=1e-6, atol=1e-7
    )
    assert fit_with_tracking.attention is None
    assert cv_with_tracking.fold_scores == cv_without_tracking.fold_scores
    assert cv_with_tracking.cv_score == cv_without_tracking.cv_score
    torch.testing.assert_close(
        cv_with_tracking.oof_predictions, cv_without_tracking.oof_predictions, rtol=1e-6, atol=1e-7
    )
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


def test_fit_leaf_regression_composes_custom_backbone_with_pgls_and_stable_subsets(
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
):
    """PGLS receives full backbone inputs and compact split-specific covariance batches."""
    from phylognn import LeafRegressionConfig, fit_leaf_regression, prepare_leaf_regression
    from phylognn.models import PGLSRegressionHead

    _LeafRepresentationBackbone.instances.clear()
    data = prepare_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
    )
    covariances = _pgls_fitting_covariances()
    pgls_loss = _RecordingMultiTraitPGLSLoss()
    pgls_head = PGLSRegressionHead(2, 2)
    result = fit_leaf_regression(
        data,
        train_indices=[5, 0, 1],
        training_config=LeafRegressionConfig(
            epochs=1,
            learning_rate=0.01,
            seed=19,
            early_stopping=True,
            early_stopping_patience=1,
        ),
        model_class=_LeafRepresentationBackbone,
        pgls_head=pgls_head,
        pgls_loss=pgls_loss,
        covariances=covariances,
        batch=_pgls_fitting_batch(),
        _tracking_validation_indices=torch.tensor([2]),
    )

    backbone = _LeafRepresentationBackbone.instances[-1]
    assert backbone.received_representations is not None
    assert backbone.received_position_mask is not None
    assert torch.equal(backbone.received_representations, data.representations)
    assert torch.equal(backbone.received_position_mask, data.position_mask)
    assert backbone.received_representations.device == result.predictions.device
    assert result.predictions.shape == (6, 2)
    assert torch.isfinite(result.predictions).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in backbone.parameters()
    )
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in pgls_head.parameters()
    )

    training_covariances, training_batch = pgls_loss.calls[0]
    assert torch.equal(training_batch, torch.tensor([0, 0, 1]))
    assert len(training_covariances) == 2
    assert torch.equal(training_covariances[0], covariances[0])
    assert torch.equal(training_covariances[1], covariances[2][1:, 1:])
    validation_covariances, validation_batch = pgls_loss.calls[1]
    assert torch.equal(validation_batch, torch.tensor([0]))
    assert len(validation_covariances) == 1
    assert torch.equal(validation_covariances[0], covariances[1][:1, :1])


@pytest.mark.parametrize("missing", ["pgls_head", "pgls_loss", "covariances", "batch"])
def test_fit_leaf_regression_requires_all_or_none_pgls_configuration(
    missing,
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
):
    """A partial explicit PGLS group fails before model construction."""
    from phylognn import LeafRegressionConfig, fit_leaf_regression, prepare_leaf_regression
    from phylognn.models import PGLSRegressionHead
    from phylognn.training import PGLSLoss

    data = prepare_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
    )
    kwargs = {
        "pgls_head": PGLSRegressionHead(2, 1),
        "pgls_loss": PGLSLoss(),
        "covariances": _pgls_fitting_covariances(),
        "batch": _pgls_fitting_batch(),
    }
    kwargs[missing] = None

    with pytest.raises(ValueError, match="all be provided or all omitted"):
        fit_leaf_regression(
            data,
            training_config=LeafRegressionConfig(epochs=1),
            model_class=_LeafRepresentationBackbone,
            **kwargs,
        )


@pytest.mark.parametrize(
    ("model_class", "message"),
    [
        (_MissingLeafRepresentationBackbone, "callable.*forward_leaf_representations"),
        (_NonCallableLeafRepresentationBackbone, "callable.*forward_leaf_representations"),
        (_IncompatibleLeafRepresentationBackbone, "incompatible.*representations.*position_mask"),
    ],
)
def test_fit_leaf_regression_rejects_invalid_pgls_backbone_protocol(
    model_class,
    message,
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
):
    from phylognn import LeafRegressionConfig, fit_leaf_regression, prepare_leaf_regression
    from phylognn.models import PGLSRegressionHead
    from phylognn.training import PGLSLoss

    data = prepare_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
    )
    with pytest.raises(TypeError, match=message):
        fit_leaf_regression(
            data,
            training_config=LeafRegressionConfig(epochs=1),
            model_class=model_class,
            pgls_head=PGLSRegressionHead(2, 1),
            pgls_loss=PGLSLoss(),
            covariances=_pgls_fitting_covariances(),
            batch=_pgls_fitting_batch(),
        )


@pytest.mark.parametrize(
    ("metadata_case", "message"),
    [
        ("empty_covariances", "covariances.*non-empty list"),
        ("batch_rank", r"batch.*shape \[N\]"),
        ("batch_dtype", "batch.*int64"),
        ("negative_batch", "batch.*non-negative"),
        ("noncontiguous_batch", "batch.*contiguous.*0..K-1"),
        ("missing_identifier", "batch.*contiguous.*0..K-1"),
        ("covariance_count", "covariance count"),
        ("leaf_count", "leaf count"),
        ("device", "covariance.*device"),
    ],
)
def test_fit_leaf_regression_rejects_invalid_pgls_metadata(
    metadata_case,
    message,
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
):
    from phylognn import LeafRegressionConfig, fit_leaf_regression, prepare_leaf_regression
    from phylognn.models import PGLSRegressionHead
    from phylognn.training import PGLSLoss

    data = prepare_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
    )
    covariances = _pgls_fitting_covariances()
    batch = _pgls_fitting_batch()
    if metadata_case == "empty_covariances":
        covariances = []
    elif metadata_case == "batch_rank":
        batch = batch.unsqueeze(1)
    elif metadata_case == "batch_dtype":
        batch = batch.to(torch.float32)
    elif metadata_case == "negative_batch":
        batch = batch - 1
    elif metadata_case == "noncontiguous_batch":
        batch = torch.tensor([0, 0, 2, 2, 3, 3])
    elif metadata_case == "missing_identifier":
        batch = torch.tensor([0, 0, 0, 0, 2, 2])
    elif metadata_case == "covariance_count":
        covariances = covariances[:2]
    elif metadata_case == "leaf_count":
        covariances[0] = torch.eye(3)
    elif metadata_case == "device":
        covariances[0] = covariances[0].to("meta")

    with pytest.raises(ValueError, match=message):
        fit_leaf_regression(
            data,
            training_config=LeafRegressionConfig(epochs=1),
            model_class=_LeafRepresentationBackbone,
            pgls_head=PGLSRegressionHead(2, 1),
            pgls_loss=PGLSLoss(),
            covariances=covariances,
            batch=batch,
        )


def test_fit_leaf_regression_pgls_disabled_preserves_scalar_output(
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
):
    """Omitting the PGLS group retains the legacy one-dimensional result contract."""
    from phylognn import LeafRegressionConfig, fit_leaf_regression, prepare_leaf_regression

    data = prepare_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
    )
    result = fit_leaf_regression(
        data,
        training_config=LeafRegressionConfig(epochs=1, seed=19),
        model_class=_ConfiguredRegressor,
        model_config={"offset": 0.0},
    )

    assert result.predictions.shape == (6,)


@pytest.mark.parametrize(
    "training_config_kwargs",
    [
        {"loss": "mse"},
        {"loss": "mae"},
        {"loss": "huber"},
    ],
)
def test_fit_leaf_regression_supports_selectable_loss_and_preserves_result_contracts(
    training_config_kwargs,
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
):
    """Every catalog loss trains with one finite loss per epoch and unchanged contracts."""
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
        training_config=LeafRegressionConfig(
            epochs=3, learning_rate=0.01, seed=19, **training_config_kwargs
        ),
        model_class=_ConfiguredRegressor,
        model_config={"offset": 0.25},
    )

    assert torch.equal(result.train_indices, torch.tensor(selected_indices))
    assert result.predictions.shape == (len(data.leaf_names),)
    assert result.attention is not None
    assert result.attention.shape == data.position_mask.shape
    assert len(result.losses) == 3
    assert all(np.isfinite(loss) for loss in result.losses)


def test_fit_leaf_regression_default_loss_matches_explicit_mse_and_direct_computation(
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
):
    """The default loss is exactly MSE, matching both an explicit selection and torch."""
    from phylognn import LeafRegressionConfig, fit_leaf_regression, prepare_leaf_regression

    data = prepare_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
    )
    selected_indices = [5, 1, 3]
    default_result = fit_leaf_regression(
        data,
        train_indices=selected_indices,
        training_config=LeafRegressionConfig(epochs=3, learning_rate=0.01, seed=19),
        model_class=_ConfiguredRegressor,
        model_config={"offset": 0.25},
    )
    explicit_mse_result = fit_leaf_regression(
        data,
        train_indices=selected_indices,
        training_config=LeafRegressionConfig(epochs=3, learning_rate=0.01, seed=19, loss="mse"),
        model_class=_ConfiguredRegressor,
        model_config={"offset": 0.25},
    )

    assert default_result.losses == explicit_mse_result.losses
    torch.testing.assert_close(default_result.predictions, explicit_mse_result.predictions)

    initial_weight = torch.tensor(0.1)
    offset = _ConfiguredRegressor.instances[-2].offset
    first_epoch_predictions = (
        leaf_regression_representations[selected_indices, 0, 0] * initial_weight + offset
    )
    expected_first_loss = torch.nn.functional.mse_loss(
        first_epoch_predictions, leaf_regression_targets[selected_indices]
    ).item()
    assert default_result.losses[0] == pytest.approx(expected_first_loss)


def test_run_leaf_regression_supports_mae_and_huber_losses_end_to_end(
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
):
    """The complete workflow finishes with finite results for non-default losses."""
    from phylognn import LeafRegressionConfig, run_leaf_regression

    for loss_name in ("mae", "huber"):
        result = run_leaf_regression(
            leaf_regression_tree,
            leaf_regression_representations,
            leaf_regression_position_mask,
            leaf_regression_targets,
            n_splits=3,
            training_config=LeafRegressionConfig(
                epochs=2, learning_rate=0.01, seed=17, loss=loss_name
            ),
        )

        assert {
            "cv_score",
            "fold_scores",
            "oof_predictions",
            "predictions",
            "attention",
            "mean_attention",
        } == set(result.__dataclass_fields__)
        assert np.isfinite(result.cv_score)
        assert len(result.fold_scores) == 3
        assert all(np.isfinite(score) for score in result.fold_scores)
        assert torch.isfinite(result.oof_predictions).all()
        assert result.oof_predictions.shape == leaf_regression_targets.shape
        assert torch.isfinite(result.predictions).all()
        assert result.predictions.shape == leaf_regression_targets.shape


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


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"loss": 42}, TypeError),
        ({"loss": "huber", "huber_delta": True}, TypeError),
        ({"loss": "huber", "huber_delta": "1.0"}, TypeError),
        ({"loss": "logcosh"}, ValueError),
        ({"loss": "MSE"}, ValueError),
        ({"loss": "huber", "huber_delta": 0}, ValueError),
        ({"loss": "huber", "huber_delta": -1.0}, ValueError),
        ({"loss": "huber", "huber_delta": float("nan")}, ValueError),
        ({"loss": "huber", "huber_delta": float("inf")}, ValueError),
        ({"loss": "mse", "huber_delta": 1.0}, ValueError),
        ({"loss": "mae", "huber_delta": 1.0}, ValueError),
    ],
)
def test_leaf_regression_config_rejects_invalid_loss_selection(kwargs, error):
    """Every invalid `loss`/`huber_delta` combination fails with its documented category."""
    from phylognn import LeafRegressionConfig

    with pytest.raises(error, match="loss.*huber_delta"):
        LeafRegressionConfig(**kwargs)


def test_leaf_regression_config_normalizes_and_defaults_huber_delta():
    """A valid integer delta normalizes to float, and an omitted delta stays unset."""
    from phylognn import LeafRegressionConfig

    normalized_config = LeafRegressionConfig(loss="huber", huber_delta=2)
    assert normalized_config.huber_delta == 2.0
    assert isinstance(normalized_config.huber_delta, float)

    omitted_config = LeafRegressionConfig(loss="huber")
    assert omitted_config.huber_delta is None


def test_leaf_regression_config_accepts_loss_selection_with_existing_defaults():
    """The default config keeps MSE, and loss selections construct cleanly."""
    from phylognn import LeafRegressionConfig

    default_config = LeafRegressionConfig()
    assert default_config.loss == "mse"
    assert default_config.huber_delta is None

    positional_config = LeafRegressionConfig(100, 1e-3, 0.0, None, None)
    assert positional_config.loss == "mse"
    assert positional_config.huber_delta is None

    mae_config = LeafRegressionConfig(loss="mae")
    assert mae_config.loss == "mae"
    assert mae_config.huber_delta is None

    huber_default_config = LeafRegressionConfig(loss="huber")
    assert huber_default_config.loss == "huber"
    assert huber_default_config.huber_delta is None

    huber_explicit_config = LeafRegressionConfig(loss="huber", huber_delta=1.5)
    assert huber_explicit_config.loss == "huber"
    assert huber_explicit_config.huber_delta == 1.5


def test_leaf_regression_config_defaults_and_appends_early_stopping_controls():
    """Early-stopping controls preserve all existing positional arguments."""
    from phylognn import LeafRegressionConfig

    default_config = LeafRegressionConfig()
    assert default_config.early_stopping is False
    assert default_config.early_stopping_patience == 20

    positional_config = LeafRegressionConfig(100, 1e-3, 0.0, None, None, "mse", None, True, 3)
    assert positional_config.early_stopping is True
    assert positional_config.early_stopping_patience == 3


@pytest.mark.parametrize("value", [1, 0, "true", None])
def test_leaf_regression_config_rejects_non_boolean_early_stopping(value):
    """The early-stopping switch accepts only exact booleans."""
    from phylognn import LeafRegressionConfig

    with pytest.raises(ValueError, match="early_stopping"):
        LeafRegressionConfig(early_stopping=value)


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "20", None])
def test_leaf_regression_config_rejects_invalid_early_stopping_patience_when_disabled(value):
    """Patience remains validated even when early stopping is disabled."""
    from phylognn import LeafRegressionConfig

    with pytest.raises(ValueError, match="early_stopping_patience"):
        LeafRegressionConfig(early_stopping=False, early_stopping_patience=value)


def test_fit_leaf_regression_rejects_enabled_early_stopping_before_tracking_or_construction(
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
):
    """Direct fits cannot enable stopping because they have no held-out leaves."""
    from phylognn import LeafRegressionConfig, fit_leaf_regression, prepare_leaf_regression
    from phylognn.training.tracking import TrackingConfig

    _ConstructionSentinelRegressor.construction_count = 0
    tracker = _ObservableLeafTracker()
    data = prepare_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
    )

    with pytest.raises(ValueError, match="early_stopping"):
        fit_leaf_regression(
            data,
            training_config=LeafRegressionConfig(early_stopping=True),
            model_class=_ConstructionSentinelRegressor,
            tracking_config=TrackingConfig(enabled=True, project="phylognn"),
            tracker=tracker,
        )

    assert _ConstructionSentinelRegressor.construction_count == 0
    assert tracker.start_payloads == []


@pytest.mark.parametrize("validation_folds", [None, ([0, 1, 2], [3, 4, 5])])
def test_early_stopping_uses_post_epoch_validation_with_strict_improvement_and_reset(
    validation_folds,
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
):
    """Equal validation losses consume patience after an improvement resets it."""
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
    targets = tuple(leaf_regression_targets.tolist())
    script = (targets, tuple(value + 1 for value in targets), targets, targets) * 2
    result = cross_validate_leaf_regression(
        data,
        n_splits=2,
        validation_folds=validation_folds,
        training_config=LeafRegressionConfig(
            epochs=8,
            learning_rate=1e-30,
            seed=5,
            early_stopping=True,
            early_stopping_patience=2,
        ),
        refit=False,
        model_class=_ScriptedLossRegressor,
        model_config={"prediction_script": script},
    )

    assert tuple(len(fold_result.losses) for fold_result in result.fold_results) == (4, 4)


def test_early_stopping_respects_patience_greater_than_epoch_limit_and_rejects_nonfinite_losses(
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
):
    """Finite folds retain their epoch cap; non-finite losses fail explicitly."""
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
    targets = tuple(leaf_regression_targets.tolist())
    stable_script = (targets,) * 12
    full_result = cross_validate_leaf_regression(
        data,
        n_splits=2,
        training_config=LeafRegressionConfig(
            epochs=3,
            learning_rate=1e-30,
            early_stopping=True,
            early_stopping_patience=4,
        ),
        refit=False,
        model_class=_ScriptedLossRegressor,
        model_config={"prediction_script": stable_script},
    )
    assert tuple(len(fold_result.losses) for fold_result in full_result.fold_results) == (3, 3)

    with pytest.raises(ValueError, match="finite"):
        cross_validate_leaf_regression(
            data,
            n_splits=2,
            training_config=LeafRegressionConfig(epochs=1, early_stopping=True),
            refit=False,
            model_class=_InvalidOutputRegressor,
            model_config={"kind": "nonfinite"},
        )

    with pytest.raises(ValueError, match="validation.*finite"):
        cross_validate_leaf_regression(
            data,
            n_splits=2,
            training_config=LeafRegressionConfig(
                epochs=1,
                learning_rate=1e-30,
                early_stopping=True,
            ),
            refit=False,
            model_class=_ScriptedLossRegressor,
            model_config={"prediction_script": (targets, (1e30,) * len(targets))},
        )


def test_early_stopping_restores_each_fold_best_state_for_predictions_attention_and_oof(
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
):
    """Fold outputs and OOF assignments use each fold's selected best state."""
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
    targets = tuple(leaf_regression_targets.tolist())
    script = (
        targets,
        targets,
        targets,
        tuple(value + 2 for value in targets),
        tuple(value + 4 for value in targets),
    )
    result = cross_validate_leaf_regression(
        data,
        validation_folds=([0, 1, 2], [3, 4, 5]),
        training_config=LeafRegressionConfig(
            epochs=6,
            learning_rate=1e-30,
            seed=7,
            early_stopping=True,
            early_stopping_patience=1,
        ),
        refit=False,
        model_class=_ScriptedLossRegressor,
        model_config={"prediction_script": script},
        score_fn=torch.nn.functional.mse_loss,
    )

    assert tuple(len(fold_result.losses) for fold_result in result.fold_results) == (2, 2)
    assert all(fold_result.attention is not None for fold_result in result.fold_results)
    torch.testing.assert_close(result.oof_predictions, leaf_regression_targets)
    assert result.fold_scores == (0.0, 0.0)
    expected_attention = leaf_regression_position_mask.to(dtype=torch.float32)
    expected_attention *= torch.arange(1, expected_attention.size(1) + 1) * 1.5 + 2
    expected_attention /= expected_attention.sum(dim=1, keepdim=True)
    for fold_result in result.fold_results:
        torch.testing.assert_close(fold_result.predictions, leaf_regression_targets)
        torch.testing.assert_close(fold_result.attention, expected_attention)


def test_early_stopping_plateau_reduces_cv_epochs_and_returns_complete_oof_without_refit(
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
):
    """A controlled plateau shortens automatic CV while retaining every OOF value."""
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
    targets = tuple(leaf_regression_targets.tolist())
    configured_epochs = 10
    result = cross_validate_leaf_regression(
        data,
        n_splits=3,
        training_config=LeafRegressionConfig(
            epochs=configured_epochs,
            learning_rate=1e-30,
            seed=13,
            early_stopping=True,
            early_stopping_patience=2,
        ),
        refit=False,
        model_class=_ScriptedLossRegressor,
        model_config={"prediction_script": (targets,) * 12},
    )

    completed_epochs = sum(len(fold_result.losses) for fold_result in result.fold_results)
    configured_total = configured_epochs * len(result.fold_results)
    assert completed_epochs <= configured_total * 0.8
    assert result.final_fit is None
    assert result.oof_predictions.shape == leaf_regression_targets.shape
    assert torch.isfinite(result.oof_predictions).all()


@pytest.mark.parametrize(
    ("fold_modes", "expected_fold_epochs"),
    [
        (("improving", "improving", "improving"), (5, 5, 5)),
        (("improving", "plateau", "improving"), (5, 2, 5)),
        (("plateau", "plateau", "plateau"), (2, 2, 2)),
    ],
)
def test_early_stopping_refit_runs_full_duration_after_any_fold_stop_pattern(
    fold_modes,
    expected_fold_epochs,
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
):
    """The final all-leaf refit never inherits a fold counter or stopping epoch."""
    from phylognn import (
        LeafRegressionConfig,
        cross_validate_leaf_regression,
        prepare_leaf_regression,
    )

    configured_epochs = 5
    targets = tuple(leaf_regression_targets.tolist())

    def _fold_script(mode):
        validation_offsets = (
            tuple(float(configured_epochs - epoch) for epoch in range(configured_epochs))
            if mode == "improving"
            else (1.0,) * configured_epochs
        )
        return tuple(
            tuple(
                target + (validation_offsets[index // 2] if index % 2 else 0.0)
                for target in targets
            )
            for index in range(configured_epochs * 2)
        )

    _RefitDurationRegressor.instances = []
    data = prepare_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
    )
    result = cross_validate_leaf_regression(
        data,
        validation_folds=([0, 1], [2, 3], [4, 5]),
        training_config=LeafRegressionConfig(
            epochs=configured_epochs,
            learning_rate=1e-30,
            early_stopping=True,
            early_stopping_patience=1,
        ),
        model_class=_RefitDurationRegressor,
        model_config={
            "stage_scripts": tuple(_fold_script(mode) for mode in fold_modes)
            + (_fold_script("improving"),)
        },
    )

    assert (
        tuple(len(fold_result.losses) for fold_result in result.fold_results)
        == expected_fold_epochs
    )
    assert result.final_fit is not None
    assert len(result.final_fit.losses) == configured_epochs


def test_run_leaf_regression_keeps_full_duration_refit_after_early_stopped_folds(
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
):
    """The recommended workflow stops folds but always completes its final refit."""
    from phylognn import LeafRegressionConfig, run_leaf_regression
    from phylognn.training.tracking import TrackingConfig

    configured_epochs = 5
    targets = tuple(leaf_regression_targets.tolist())
    plateau_script = tuple(
        tuple(target + (1.0 if index % 2 else 0.0) for target in targets)
        for index in range(configured_epochs * 2)
    )
    tracker = _ObservableLeafTracker()
    _RefitDurationRegressor.instances = []
    _RefitDurationRegressor.configured_stage_scripts = (plateau_script,) * 4

    result = run_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
        validation_folds=([0, 1], [2, 3], [4, 5]),
        training_config=LeafRegressionConfig(
            epochs=configured_epochs,
            learning_rate=1e-30,
            early_stopping=True,
            early_stopping_patience=1,
        ),
        model_class=_RefitDurationRegressor,
        tracking_config=TrackingConfig(enabled=True, project="phylognn"),
        tracker=tracker,
    )

    for fold_index in range(1, 4):
        tracker.assert_stage_epochs("cv_fold", fold_index, [1, 2])
    tracker.assert_stage_epochs("refit", 1, list(range(1, configured_epochs + 1)))
    assert result.predictions.shape == leaf_regression_targets.shape


@pytest.mark.parametrize("explicitly_disabled", [False, True])
def test_early_stopping_omitted_or_disabled_preserves_fixed_epoch_baselines(
    explicitly_disabled,
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
):
    """Disabled stopping retains pre-feature results across every public workflow."""
    from phylognn import (
        LeafRegressionConfig,
        cross_validate_leaf_regression,
        fit_leaf_regression,
        prepare_leaf_regression,
        run_leaf_regression,
    )

    baseline_config = _FIXED_EPOCH_BASELINE["config"]
    config_kwargs = {
        "epochs": baseline_config["epochs"],
        "learning_rate": baseline_config["learning_rate"],
        "seed": baseline_config["seed"],
    }
    if explicitly_disabled:
        config_kwargs["early_stopping"] = False
    config = LeafRegressionConfig(**config_kwargs)
    model_kwargs = {
        "model_class": _ConfiguredRegressor,
        "model_config": {"offset": baseline_config["offset"]},
    }
    data = prepare_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
    )
    expected_attention = leaf_regression_position_mask.to(dtype=torch.float32)
    expected_attention /= expected_attention.sum(dim=1, keepdim=True)

    fit_result = fit_leaf_regression(data, training_config=config, **model_kwargs)
    assert fit_result.losses == pytest.approx(_FIXED_EPOCH_BASELINE["fit"]["losses"])
    torch.testing.assert_close(
        fit_result.predictions,
        torch.tensor(_FIXED_EPOCH_BASELINE["fit"]["predictions"]),
    )
    torch.testing.assert_close(fit_result.attention, expected_attention)

    manual_result = cross_validate_leaf_regression(
        data,
        validation_folds=([0, 1], [2, 3], [4, 5]),
        training_config=config,
        score_fn=torch.nn.functional.mse_loss,
        refit=False,
        **model_kwargs,
    )
    manual_baseline = _FIXED_EPOCH_BASELINE["manual_cv"]
    for result, expected_losses in zip(
        manual_result.fold_results, manual_baseline["fold_losses"], strict=True
    ):
        assert result.losses == pytest.approx(expected_losses)
        torch.testing.assert_close(result.attention, expected_attention)
    assert manual_result.fold_scores == pytest.approx(manual_baseline["fold_scores"])
    assert manual_result.cv_score == pytest.approx(manual_baseline["cv_score"])
    torch.testing.assert_close(
        manual_result.oof_predictions,
        torch.tensor(manual_baseline["oof_predictions"]),
    )

    automatic_result = cross_validate_leaf_regression(
        data,
        n_splits=3,
        training_config=config,
        score_fn=torch.nn.functional.mse_loss,
        refit=False,
        **model_kwargs,
    )
    automatic_baseline = _FIXED_EPOCH_BASELINE["automatic_cv"]
    assert tuple(tuple(fold.tolist()) for fold in automatic_result.validation_folds) == (
        automatic_baseline["validation_folds"]
    )
    for result, expected_losses in zip(
        automatic_result.fold_results, automatic_baseline["fold_losses"], strict=True
    ):
        assert result.losses == pytest.approx(expected_losses)
        torch.testing.assert_close(result.attention, expected_attention)
    assert automatic_result.fold_scores == pytest.approx(automatic_baseline["fold_scores"])
    assert automatic_result.cv_score == pytest.approx(automatic_baseline["cv_score"])
    torch.testing.assert_close(
        automatic_result.oof_predictions,
        torch.tensor(automatic_baseline["oof_predictions"]),
    )

    workflow_result = run_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
        validation_folds=([0, 1], [2, 3], [4, 5]),
        training_config=config,
        score_fn=torch.nn.functional.mse_loss,
        **model_kwargs,
    )
    workflow_baseline = _FIXED_EPOCH_BASELINE["workflow"]
    assert workflow_result.cv_score == pytest.approx(workflow_baseline["cv_score"])
    assert workflow_result.fold_scores == pytest.approx(manual_baseline["fold_scores"])
    torch.testing.assert_close(
        workflow_result.oof_predictions,
        torch.tensor(manual_baseline["oof_predictions"]),
    )
    torch.testing.assert_close(
        workflow_result.predictions,
        torch.tensor(workflow_baseline["predictions"]),
    )
    torch.testing.assert_close(workflow_result.attention, expected_attention)
    torch.testing.assert_close(workflow_result.mean_attention, expected_attention.mean(dim=0))


def test_early_stopping_continuously_improving_folds_complete_all_epochs(
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
):
    """Strictly improving held-out losses never consume early-stopping patience."""
    from phylognn import (
        LeafRegressionConfig,
        cross_validate_leaf_regression,
        prepare_leaf_regression,
    )

    configured_epochs = 4
    targets = tuple(leaf_regression_targets.tolist())
    prediction_script = tuple(
        (
            targets
            if index % 2 == 0
            else tuple(target + configured_epochs - index // 2 for target in targets)
        )
        for index in range(configured_epochs * 2)
    )
    data = prepare_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
    )

    result = cross_validate_leaf_regression(
        data,
        validation_folds=([0, 1, 2], [3, 4, 5]),
        training_config=LeafRegressionConfig(
            epochs=configured_epochs,
            learning_rate=1e-30,
            early_stopping=True,
            early_stopping_patience=1,
        ),
        refit=False,
        model_class=_ScriptedLossRegressor,
        model_config={"prediction_script": prediction_script},
    )

    assert tuple(len(fold_result.losses) for fold_result in result.fold_results) == (
        configured_epochs,
        configured_epochs,
    )


def test_early_stopping_tracking_records_config_and_actual_stage_epochs(
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
):
    """All tracked entry points expose stopping config and completed stage durations."""
    from phylognn import (
        LeafRegressionConfig,
        cross_validate_leaf_regression,
        fit_leaf_regression,
        prepare_leaf_regression,
        run_leaf_regression,
    )
    from phylognn.training.tracking import TrackingConfig

    configured_epochs = 4
    targets = tuple(leaf_regression_targets.tolist())
    plateau_script = (targets,) * (configured_epochs * 2)
    data = prepare_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
    )
    tracking_config = TrackingConfig(enabled=True, project="phylognn")

    direct_tracker = _ObservableLeafTracker()
    _RefitDurationRegressor.instances = []
    _RefitDurationRegressor.configured_stage_scripts = (plateau_script,)
    fit_leaf_regression(
        data,
        training_config=LeafRegressionConfig(
            epochs=configured_epochs,
            early_stopping_patience=3,
        ),
        model_class=_RefitDurationRegressor,
        tracking_config=tracking_config,
        tracker=direct_tracker,
    )
    assert direct_tracker.start_payloads[0]["training.early_stopping"] is False
    assert direct_tracker.start_payloads[0]["training.early_stopping_patience"] == 3

    cross_validation_tracker = _ObservableLeafTracker()
    _RefitDurationRegressor.instances = []
    _RefitDurationRegressor.configured_stage_scripts = (plateau_script,) * 3
    cross_validate_leaf_regression(
        data,
        validation_folds=([0, 1], [2, 3], [4, 5]),
        training_config=LeafRegressionConfig(
            epochs=configured_epochs,
            learning_rate=1e-30,
            early_stopping=True,
            early_stopping_patience=1,
        ),
        score_fn=torch.nn.functional.mse_loss,
        refit=False,
        model_class=_RefitDurationRegressor,
        tracking_config=tracking_config,
        tracker=cross_validation_tracker,
    )
    assert cross_validation_tracker.start_payloads[0]["training.early_stopping"] is True
    assert cross_validation_tracker.start_payloads[0]["training.early_stopping_patience"] == 1
    for fold_index in range(1, 4):
        cross_validation_tracker.assert_stage_epochs("cv_fold", fold_index, [1, 2])

    workflow_tracker = _ObservableLeafTracker()
    _RefitDurationRegressor.instances = []
    _RefitDurationRegressor.configured_stage_scripts = (plateau_script,) * 4
    run_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
        validation_folds=([0, 1], [2, 3], [4, 5]),
        training_config=LeafRegressionConfig(
            epochs=configured_epochs,
            learning_rate=1e-30,
            early_stopping=True,
            early_stopping_patience=1,
        ),
        score_fn=torch.nn.functional.mse_loss,
        model_class=_RefitDurationRegressor,
        tracking_config=tracking_config,
        tracker=workflow_tracker,
    )
    assert workflow_tracker.start_payloads[0]["training.early_stopping"] is True
    assert workflow_tracker.start_payloads[0]["training.early_stopping_patience"] == 1
    for fold_index in range(1, 4):
        workflow_tracker.assert_stage_epochs("cv_fold", fold_index, [1, 2])
    workflow_tracker.assert_stage_epochs("refit", 1, list(range(1, configured_epochs + 1)))


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


# ---------------------------------------------------------------------------
# T003: Default-model workflow test with entmax15 attention normalization
# ---------------------------------------------------------------------------


def test_default_model_entmax15_workflow(
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
    torch_module,
):
    """model_config forwards attention_normalization='entmax15' to the default model."""
    from phylognn import LeafRegressionConfig, fit_leaf_regression, prepare_leaf_regression

    data = prepare_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
    )
    result = fit_leaf_regression(
        data,
        training_config=LeafRegressionConfig(epochs=2, learning_rate=0.01, seed=42),
        model_config={"attention_normalization": "entmax15"},
    )

    assert result.predictions.shape == (6,)
    assert torch_module.isfinite(result.predictions).all()
    assert result.attention is not None
    assert result.attention.shape == (6, 4)
    assert torch_module.isfinite(result.attention).all()
    # Attention must be non-negative
    assert (result.attention >= 0).all()
    # Masked positions must be exactly zero
    mask = leaf_regression_position_mask.to(dtype=torch_module.bool)
    assert torch_module.equal(
        result.attention[~mask],
        torch_module.zeros_like(result.attention[~mask]),
    )
    # Each row sums to 1 within tolerance
    assert torch_module.allclose(
        result.attention.sum(dim=1), torch_module.ones(6), atol=1e-6, rtol=0
    )


@pytest.mark.parametrize("model_name", ["default", "sparse_query"])
def test_fit_leaf_regression_forwards_chunk_size_to_selected_sequence_model(
    model_name,
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
):
    """The existing model configuration forwards chunking without result changes."""
    from phylognn import LeafRegressionConfig, fit_leaf_regression, prepare_leaf_regression
    from phylognn.leaf_regression.fitting import _construct_model
    from phylognn.models import SparseQueryPhyloRegressor

    data = prepare_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
    )
    model_class = None
    model_config: dict[str, object] = {"chunk_size": 2, "dropout_prob": 0.0}
    if model_name == "sparse_query":
        model_class = SparseQueryPhyloRegressor
        model_config = {
            "input_dim": data.representations.size(-1),
            "leaf_laplacian": data.leaf_laplacian,
            "adapter_rank": 3,
            "token_dim": 4,
            "num_cnn_blocks": 1,
            "cnn_kernel_sizes": (3,),
            "num_queries": 2,
            "slot_dim": 3,
            "species_dim": 5,
            "sequence_hidden_dim": 3,
            "phylogeny_hidden_dim": 3,
            "adapter_dropout_prob": 0.0,
            "cnn_dropout_prob": 0.0,
            "representation_dropout_prob": 0.0,
            "sequence_dropout_prob": 0.0,
            "phylogeny_dropout_prob": 0.0,
            "chunk_size": 2,
        }

    model = _construct_model(data, model_class, model_config)
    assert model.chunk_size == 2

    if model_name == "default":
        result = fit_leaf_regression(
            data,
            training_config=LeafRegressionConfig(epochs=1, learning_rate=0.01, seed=42),
            model_config=model_config,
        )

        assert result.predictions.shape == (6,)
        assert result.attention is not None
        assert result.attention.shape[0] == 6
        assert len(result.losses) == 1


class _RepresentationStorageRegressor(torch.nn.Module):
    """Record the representation tensor received by one fitting call."""

    instances: list["_RepresentationStorageRegressor"] = []

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.1))
        self.representation_data_ptr: int | None = None
        self.representation_version: int | None = None
        type(self).instances.append(self)

    def forward(self, representations: torch.Tensor, position_mask: torch.Tensor) -> torch.Tensor:
        del position_mask
        self.representation_data_ptr = representations.data_ptr()
        self.representation_version = representations._version
        return representations[:, 0, 0] * self.weight


def _snapshot_leaf_regression_data(data):
    """Capture caller-owned values and representation storage metadata."""
    return {
        "representations": data.representations.detach().clone(),
        "representation_data_ptr": data.representations.data_ptr(),
        "representation_version": data.representations._version,
        "position_mask": data.position_mask.detach().clone(),
        "targets": data.targets.detach().clone(),
        "leaf_laplacian": data.leaf_laplacian.detach().clone(),
    }


def _assert_leaf_regression_data_matches_snapshot(data, snapshot):
    """Assert fitting did not mutate any caller-owned data field."""
    assert torch.equal(data.representations, snapshot["representations"])
    assert data.representations.data_ptr() == snapshot["representation_data_ptr"]
    assert data.representations._version == snapshot["representation_version"]
    assert torch.equal(data.position_mask, snapshot["position_mask"])
    assert torch.equal(data.targets, snapshot["targets"])
    assert torch.equal(data.leaf_laplacian, snapshot["leaf_laplacian"])


def test_fit_leaf_regression_uses_same_storage_representation_alias_on_target_device(
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
):
    """A same-device fit avoids a redundant representations clone."""
    from phylognn import LeafRegressionConfig, fit_leaf_regression, prepare_leaf_regression

    _RepresentationStorageRegressor.instances.clear()
    data = prepare_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
    )
    snapshot = _snapshot_leaf_regression_data(data)
    train_indices = torch.tensor([0, 2, 4], dtype=torch.long)
    indices_snapshot = train_indices.detach().clone()
    indices_data_ptr = train_indices.data_ptr()
    indices_version = train_indices._version

    fit_leaf_regression(
        data,
        train_indices=train_indices,
        training_config=LeafRegressionConfig(epochs=1, learning_rate=0.01, seed=17),
        model_class=_RepresentationStorageRegressor,
    )

    model = _RepresentationStorageRegressor.instances[-1]
    assert model.representation_data_ptr == snapshot["representation_data_ptr"]
    assert model.representation_version == snapshot["representation_version"]
    _assert_leaf_regression_data_matches_snapshot(data, snapshot)
    assert torch.equal(train_indices, indices_snapshot)
    assert train_indices.data_ptr() == indices_data_ptr
    assert train_indices._version == indices_version


def test_prepare_representations_detaches_on_target_device_and_transfers_otherwise():
    """Representation preparation aliases locally and accepts transfer results remotely."""
    from phylognn.leaf_regression.fitting import _prepare_representations

    representations = torch.ones((2, 3, 4), dtype=torch.float32)

    local = _prepare_representations(representations, torch.device("cpu"))
    transferred = _prepare_representations(representations, torch.device("meta"))

    assert local.data_ptr() == representations.data_ptr()
    assert local._version == representations._version
    assert local.requires_grad is False
    assert transferred.device.type == "meta"
    assert transferred is not representations


@pytest.mark.parametrize("failure", ["invalid_indices", "model_output"])
def test_fit_leaf_regression_preserves_caller_data_after_validation_failure(
    failure,
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
):
    """Pre-update validation failures cannot mutate caller-owned fit inputs."""
    from phylognn import LeafRegressionConfig, fit_leaf_regression, prepare_leaf_regression

    data = prepare_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
    )
    snapshot = _snapshot_leaf_regression_data(data)
    kwargs = {
        "training_config": LeafRegressionConfig(epochs=1, learning_rate=0.01, seed=17),
    }
    if failure == "invalid_indices":
        kwargs["train_indices"] = [len(data.leaf_names)]
        expected_error = "train_indices"
    else:
        kwargs["model_class"] = _InvalidOutputRegressor
        kwargs["model_config"] = {"kind": "shape"}
        expected_error = "predictions"

    with pytest.raises((TypeError, ValueError), match=expected_error):
        fit_leaf_regression(data, **kwargs)

    _assert_leaf_regression_data_matches_snapshot(data, snapshot)


def test_fit_leaf_regression_keeps_default_sequence_model_configuration_compatible(
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
):
    """Omitting chunk_size retains the default full-batch sequence-model path."""
    from phylognn import LeafRegressionConfig, fit_leaf_regression, prepare_leaf_regression
    from phylognn.leaf_regression.fitting import _construct_model

    data = prepare_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
    )
    model = _construct_model(data, None, {"dropout_prob": 0.0})
    result = fit_leaf_regression(
        data,
        training_config=LeafRegressionConfig(epochs=1, learning_rate=0.01, seed=17),
        model_config={"dropout_prob": 0.0},
    )

    assert model.chunk_size is None
    assert result.predictions.shape == (len(data.leaf_names),)
    assert result.attention is not None
    assert result.attention.shape == data.position_mask.shape


@pytest.mark.parametrize(
    "loss",
    ["mse", "mae"],
)
def test_leaf_regression_config_mismatch_error_identifies_huber(loss):
    """The error for a non-Huber loss supplied with huber_delta names Huber."""
    from phylognn import LeafRegressionConfig

    with pytest.raises(ValueError, match=r"does not accept parameters.*accepted by.*huber"):
        LeafRegressionConfig(loss=loss, huber_delta=1.0)
