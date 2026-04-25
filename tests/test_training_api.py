"""Tests for the curated training package export contract."""

import importlib


def test_training_package_uses_canonical_all_exports():
    """The training package should publish a trustworthy public API."""
    training = importlib.import_module("phylognn.training")

    assert training.__all__ == [
        "ConfiguredTrainingSetup",
        "DatasetSplit",
        "SplitDatasetView",
        "SplitPhyloDataset",
        "SplitPhyloDiskDataset",
        "Trainer",
        "TrainingConfigError",
        "TrainingConfig",
        "TrackingConfig",
        "TrackingError",
        "TrackingRunInfo",
        "WandbTracker",
        "create_default_trainer",
        "create_trainer_from_config",
        "load_training_config",
        "mse_metric",
        "mae_metric",
        "r2_metric",
        "rmse_metric",
        "relative_error_metric",
    ]
    assert not hasattr(training, "all")


def test_training_package_exports_intended_metrics_and_factory_names():
    """The curated public contract should include metrics and the trainer factory."""
    training = importlib.import_module("phylognn.training")

    for export_name in {
        "create_default_trainer",
        "create_trainer_from_config",
        "load_training_config",
        "rmse_metric",
        "relative_error_metric",
        "TrackingConfig",
        "TrackingError",
        "TrackingRunInfo",
        "WandbTracker",
    }:
        assert export_name in training.__all__


def test_training_package_dir_matches_curated_api():
    """Directory listings should expose the same curated training helpers."""
    training = importlib.import_module("phylognn.training")

    for export_name in training.__all__:
        assert export_name in dir(training)


def test_training_package_rejects_unknown_attributes():
    """Unknown training facade attributes should fail loudly."""
    training = importlib.import_module("phylognn.training")

    try:
        getattr(training, "not_a_real_training_symbol")
    except AttributeError:
        pass
    else:
        raise AssertionError("Training facade unexpectedly exposed an unknown symbol.")
