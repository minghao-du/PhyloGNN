Training Reference
==================

Import path
-----------

.. code-block:: python

   from phylognn.training import TrainingConfig, Trainer, load_training_config

Configuration and trainer
-------------------------

.. py:class:: TrainingConfig

   Dataclass controlling epochs, batch size, learning rate, optimizer,
   scheduler, device, checkpoint directory, early stopping, gradient clipping,
   and loader options.

   .. py:method:: validate()

      Raise `ValueError` when a setting is outside the supported range.

.. py:class:: Trainer(model, config, loss_fn=None, metrics=None, tracking_config=None, tracking_metadata=None, tracker=None)

   Local trainer for PyG datasets and loaders. Samples must expose targets as
   `data.y`.

   .. py:method:: fit(train_loader=None, val_loader=None, train_dataset=None, val_dataset=None)

      Train the model and update local history and checkpoints.

   .. py:method:: predict(loader=None, dataset=None, return_targets=False)

      Run inference from a loader or dataset.

   .. py:method:: save_checkpoint(filename)

      Save training state under `config.save_dir`.

   .. py:method:: load_checkpoint(filename)

      Restore model, optimizer, scheduler, and history state from a trusted
      checkpoint file.

   .. py:method:: save_history(filename="history.json")

      Write local training history as JSON.

.. py:exception:: TrainingConfigError

   Raised when a `TOML` training configuration cannot be parsed or validated.

.. py:class:: ConfiguredTrainingSetup

   Effective setup returned by `load_training_config()`: model, training
   config, loss function, metrics, tracking config, and sanitized tracking
   metadata.

Datasets and splits
-------------------

.. py:class:: DatasetSplit

   Deterministic train, validation, and test split definition.

   .. py:classmethod:: from_ratios(sample_ids, train_ratio, val_ratio, test_ratio, seed=42, shuffle=True)

      Build a split from ratios.

   .. py:classmethod:: from_dict(splits)

      Build a split from explicit sample IDs.

   .. py:classmethod:: from_manifest_dir(manifest_dir)

      Load `train.txt`, `val.txt`, and `test.txt` manifests.

.. py:class:: SplitDatasetView

   Lightweight view over a split-specific subset of a base dataset.

.. py:class:: SplitPhyloDataset(data_list, labels=None, sample_ids=None, transform=None, pre_transform=None)

   In-memory dataset for graph objects and labels.

.. py:class:: SplitPhyloDiskDataset(graph_dir, label_dir=None, sample_ids=None, recursive=False, cache=True, transform=None, pre_transform=None)

   Disk-backed dataset for trusted mirrored graph and label `.pt` files.

TOML helpers
------------

.. py:function:: load_training_config(path, *, model_overrides=None, training_overrides=None, loss=None, metrics=None)

   Load a TOML file and return `ConfiguredTrainingSetup`.

.. py:function:: create_trainer_from_config(path, *, model_overrides=None, training_overrides=None, loss=None, metrics=None)

   Create a `Trainer` from a TOML configuration.

.. py:function:: create_default_trainer(model, **kwargs)

   Create a trainer with default local settings.

Metrics
-------

Trainer metrics are configured with supported string keys or direct
`torchmetrics.Metric` instances. Built-in keys are:

.. list-table::
   :header-rows: 1

   * - Key
     - Metric
   * - `mse`
     - Mean squared error
   * - `mae`
     - Mean absolute error
   * - `rmse`
     - Root mean squared error
   * - `r2`
     - R-squared score for scalar outputs by default
   * - `mape`
     - Mean absolute percentage error

The legacy helper functions `mse_metric`, `mae_metric`, `rmse_metric`,
`r2_metric`, and `relative_error_metric` are no longer part of the public API.
Use the keys above or instantiate TorchMetrics classes directly.

Tracking
--------

.. py:class:: phylognn.training.tracking.TrackingConfig

   Optional experiment tracking settings. Disabled tracking imports no external
   backend.

.. py:class:: phylognn.training.tracking.TrackerProtocol

   Protocol for an enabled tracking backend injected into a training entry
   point.

.. py:class:: phylognn.training.tracking.TrackingRunInfo

   External run identity returned after tracking starts.

.. py:exception:: phylognn.training.tracking.TrackingError

   Raised when tracking configuration or logging fails.

.. py:class:: phylognn.training.tracking.WandbTracker(tracking_config)

   Weights & Biases adapter with lazy `wandb` import.

Configuration sections and outputs
----------------------------------

`TOML` files use `[model]`, `[model.params]`, `[training]`, optional `[loss]`,
optional `[metrics]`, and optional `[tracking]`. Training writes local
checkpoints and `history.json`; enabled tracking logs sanitized run metadata
and metrics externally.

Related guide
-------------

See :doc:`../user_guide/training_config`, :doc:`../user_guide/training`, and
:doc:`../user_guide/metrics_tracking`.
