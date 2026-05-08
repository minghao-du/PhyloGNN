Training Configuration
======================

Training configuration files are local `TOML` documents loaded with Python
stdlib `tomllib`. They configure model construction, trainer settings, loss,
metrics, and optional tracking. They do not construct datasets or data loaders.

Required sections
-----------------

`[model]`
   Contains `type`, currently `GATBiLSTMNet`, and the nested
   `[model.params]` table.

`[model.params]`
   Contains model constructor values. `input_dim` and `output_dim` are
   required. Other accepted keys include GAT dimensions, temporal mode,
   number of time bins, pooling, and output-head settings.

`[training]`
   Contains `TrainingConfig` values such as `epochs`, `batch_size`,
   `learning_rate`, `optimizer`, `scheduler`, `device`, and checkpoint
   settings.

Optional sections
-----------------

`[loss]`
   Selects a built-in loss by `name`. Supported names are `mse` and `mae`.

`[metrics]`
   Selects built-in metrics by `names`, including `mse`, `mae`, `rmse`, `r2`,
   and `relative_error`.

`[tracking]`
   Enables optional experiment tracking. When `enabled=true`, `project` is
   required and the `wandb` extra must be installed.

Validation expectations
-----------------------

Unknown top-level keys, unknown section keys, missing required fields, invalid
types, unsupported model types, and invalid trainer values raise
`TrainingConfigError`. Tracking validation raises `TrackingError` when enabled
tracking lacks required settings or dependencies.

Training outputs
----------------

The trainer writes checkpoints and `history.json` under `TrainingConfig.save_dir`.
External tracking stores sanitized configuration, epoch metrics, final metrics,
and terminal status only when enabled.

See :doc:`../user_guide/training`, :doc:`../user_guide/metrics_tracking`, and
:doc:`../reference/training` for workflow and API details.
