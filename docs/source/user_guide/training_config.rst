Training Configuration
======================

Training configuration files are local `TOML` documents loaded with Python's
standard `tomllib` reader. They configure model construction, trainer settings,
loss, metrics, and optional tracking. They do not construct datasets or data
loaders.

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
   and `mape`.

`[tracking]`
   Enables optional experiment tracking. When `enabled=true`, `project` is
   required and the `wandb` extra must be installed. Its optional `metrics`
   array selects quantitative tracking metrics.

Tracking metric selection
-------------------------

Use ``[tracking].metrics`` to select the quantitative values sent to an enabled
W&B run:

.. code-block:: toml

   [tracking]
   enabled = true
   project = "phylognn"
   metrics = ["train/loss", "val/loss"]

Omit ``metrics`` to record every applicable quantitative metric. Set
``metrics = []`` to omit all quantitative metrics while retaining sanitized
configuration, stage identity, steps, and terminal status. The array is
converted to the immutable ``TrackingConfig.metrics`` tuple used by Python
callers.

Fixed names are listed in :doc:`metrics_tracking`. A standard trainer may also
select ``train/<name>`` or ``val/<name>`` for a configured trainer metric.
Names that are unavailable for the active event are omitted rather than
fabricated.

Validation expectations
-----------------------

Unknown top-level keys, unknown section keys, missing required fields, invalid
types, unsupported model types, and invalid trainer values raise
`TrainingConfigError`. ``tracking.metrics`` must be a TOML array of strings;
duplicates, malformed names, secret-like namespace segments, unknown fixed
names, and unconfigured dynamic names fail before an enabled tracker starts.
Tracking validation raises `TrackingError` when enabled tracking lacks required
settings or dependencies.

Training outputs
----------------

The trainer writes checkpoints and `history.json` under
`TrainingConfig.save_dir`. External tracking stores sanitized configuration,
epoch metrics, final metrics, and terminal status only when enabled.

Related pages
-------------

See :doc:`datasets_and_splits`, :doc:`training`, :doc:`metrics_tracking`, and
:doc:`../reference/training`.
