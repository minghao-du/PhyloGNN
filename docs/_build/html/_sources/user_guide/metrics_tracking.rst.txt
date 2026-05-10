Metrics And Tracking
====================

PhyloGNN provides built-in metrics and optional Weights & Biases tracking for
training runs.

Metrics
-------

Trainer metrics use TorchMetrics stateful objects. Pass built-in string keys
such as `mse`, `mae`, `rmse`, `r2`, and `mape`, or pass an instantiated
`torchmetrics.Metric` object. `TOML` configuration selects the same built-in
keys in the `[metrics]` section.

.. code-block:: python

   trainer = Trainer(model=model, config=config, metrics={"rmse": "rmse"})

Multi-output R2 requires a TorchMetrics `R2Score` instance configured for the
model output shape, or an explicit internal configuration path that sets a
positive output count. Direct custom metrics must inherit from
`torchmetrics.Metric`.

Local-first behavior
--------------------

Training remains local when tracking is disabled. Enable tracking only when a
run should send sanitized configuration, metrics, and status to an external
`wandb` run.

Tracking configuration
----------------------

`TrackingConfig(enabled=False)` keeps training local and does not import
wandb. When tracking is enabled, the backend must be `wandb`, `project` is
required, and the `wandb` extra must be installed.

Tracking choices
----------------

.. list-table::
   :header-rows: 1

   * - Need
     - Setting
     - Result
   * - Local development
     - `TrackingConfig(enabled=False)`
     - No external import or network logging.
   * - Logged experiment
     - `TrackingConfig(enabled=True, project="...")`
     - Logs sanitized config, metrics, and status to W&B.
   * - TOML-driven tracking
     - `[tracking] enabled = true`
     - Requires `project` and the `wandb` extra.

External run metadata
---------------------

The trainer logs sanitized configuration metadata, epoch metrics, final
metrics, and terminal status. Metadata keys that look like secrets are
rejected. Path-like metadata values are reduced to their final file or
directory names before logging.

Common failures
---------------

`TrackingError` is raised for invalid tracking settings, missing wandb, unsafe
metadata, invalid metric payloads, and backend logging failures. Disable
tracking to keep the workflow local while debugging model or data issues.

Related pages
-------------

See :doc:`training`, :doc:`training_config`, :doc:`../reference/training`, and
:doc:`../troubleshooting`.
