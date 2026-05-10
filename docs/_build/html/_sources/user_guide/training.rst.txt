Training
========

Training utilities expect PyTorch Geometric `Data` samples whose target is
stored as `data.y`. Datasets, splits, and loaders are caller-controlled; `TOML`
configuration builds models and trainer settings but does not discover data.

Trainer setup
-------------

`TrainingConfig` controls epochs, batch size, optimizer, scheduler, device,
checkpoint directory, early stopping, gradient clipping, and `DataLoader`
options. `Trainer` validates the config, creates the optimizer and scheduler,
writes `config.json`, stores training history, and saves checkpoints.

TOML setup
----------

`load_training_config()` returns a `ConfiguredTrainingSetup` containing a
`GATBiLSTMNet`, `TrainingConfig`, loss, metrics, tracking configuration, and
sanitized tracking metadata. `create_trainer_from_config()` creates a
`Trainer` from the same TOML file.

Model data contract
-------------------

Model inputs follow the graph contract in :doc:`graph_conversion`. Batched
graph-level training needs `data.batch`. Temporal modes of `GATBiLSTMNet`
require `data.time_bin` as a node-aligned tensor. The training dataset must
provide `data.y` for loss computation.

Outputs
-------

Training writes checkpoints and `history.json` under `save_dir`. The final
checkpoint is always saved; best-checkpoint behavior is controlled by
`save_best_only`.

Related pages
-------------

See :doc:`datasets_and_splits`, :doc:`training_config`, :doc:`metrics_tracking`,
:doc:`../reference/training`, and :doc:`../troubleshooting`.
