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

Metric catalog
--------------

Tracking sends only finite scalar values. The fixed quantitative catalog is
ordered as follows; a metric that is not applicable to a particular workflow
or event is omitted.

.. list-table::
   :header-rows: 1

   * - Name
     - Meaning
   * - ``train/loss``
     - Training loss for a completed epoch.
   * - ``train/score``
     - Configured regression score calculated from all training predictions and targets in a completed leaf-regression epoch.
   * - ``train/mae``
     - Mean absolute error calculated from all training predictions and targets in a completed leaf-regression epoch.
   * - ``train/pearson_r``
     - Defined Pearson correlation for all training predictions and targets in a completed leaf-regression epoch.
   * - ``train/lr``
     - Current optimizer learning rate.
   * - ``train/epoch_time_sec``
     - Completed epoch duration in seconds.
   * - ``val/loss``
     - Validation loss when validation is available.
   * - ``val/score``
     - Configured regression score calculated from all validation predictions and targets in a completed leaf-regression epoch.
   * - ``val/mae``
     - Mean absolute error calculated from all validation predictions and targets in a completed leaf-regression epoch.
   * - ``val/pearson_r``
     - Defined Pearson correlation for all validation predictions and targets in a completed leaf-regression epoch.
   * - ``final/best_val_loss``
     - Best finite validation loss.
   * - ``final/best_epoch``
     - Epoch of the best finite validation loss.
   * - ``cv/fold_score``
     - Configured score for a held-out leaf fold.
   * - ``cv/validation_leaf_count``
     - Number of held-out leaves in that fold.
   * - ``cv/mean_score``
     - Arithmetic mean of fold scores.
   * - ``cv/weighted_score``
     - Fold score weighted by held-out leaf count.
   * - ``cv/std_score``
     - Population standard deviation of fold scores.
   * - ``cv/min_score``
     - Minimum fold score.
   * - ``cv/max_score``
     - Maximum fold score.
   * - ``cv/mae``
     - Mean absolute error for aligned out-of-fold predictions and targets.
   * - ``cv/pearson_r``
     - Pearson correlation for aligned out-of-fold predictions and targets.

For each metric configured on a standard ``Trainer``, ``train/<name>`` and
``val/<name>`` are also accepted. They are emitted only when that trainer has
the named metric and the corresponding training or validation value exists.
The legacy top-level ``lr`` and ``epoch_time_sec`` fields are not emitted.

Metric selection
----------------

``TrackingConfig.metrics`` controls quantitative fields only. Its three states
are explicit:

.. code-block:: python

   # Record every applicable quantitative metric (the default).
   TrackingConfig(metrics=None)

   # Record no quantitative metrics.
   TrackingConfig(metrics=())

   # Record only applicable names in this ordered allowlist.
   TrackingConfig(metrics=("train/loss", "train/score", "val/loss", "val/score"))

An allowlist may include fixed catalog names and, for a standard trainer,
configured dynamic ``train/<name>`` and ``val/<name>`` names. A valid name for
another workflow is simply absent from events where it does not apply. For leaf
regression, ``train/score``, ``train/mae``, ``train/pearson_r`` and their
``val/`` equivalents select complete-partition epoch values. See
:doc:`training_config` for the equivalent TOML syntax.

Operational fields
------------------

Metric selection never removes sanitized run configuration, ``stage/type``,
``stage/index``, ``stage/epoch``, the backend step, or ``status/state``. Thus
an empty selection still leaves a useful stage and lifecycle record. A
successfully started run records exactly one terminal ``status/state`` value:
``completed``, ``failed``, or ``interrupted``.

Selection validation
--------------------

For enabled tracking, selection is validated before the tracker starts.
Selections must be ``None`` or a tuple of strings in Python, cannot contain
duplicates, and must use known fixed or configured dynamic names. Empty,
malformed, unknown, or sensitive namespace segments fail with ``TrackingError``.
Sensitive segments include case-insensitive forms containing ``api_key``,
``apikey``, ``auth``, ``credential``, ``password``, ``secret``, or ``token``.
Disabled tracking remains inert: it does not import W&B or validate a workflow
selection.

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
directory names before logging. Raw graphs, trees, tensors, predictions,
attention, model state, checkpoints, and artifacts are not uploaded.

Leaf-regression scalar boundary
--------------------------------

Leaf-regression runs use the same boundary. A tracked workflow sends a
sanitized scalar configuration and scalar events such as:

.. code-block:: text

   workflow.type = "run_leaf_regression"
   data.leaf_count = 24
   stage/type = "cv_fold"
   stage/index = 1
   train/loss = 0.418
   train/score = 0.731
   train/mae = 0.216
   train/pearson_r = 0.892
   cv/fold_score = 0.731
   status/state = "completed"

Trees, leaf names or index lists, tensors, predictions, attention, model
state, checkpoints, and artifacts are never uploaded. Tracking remains
disabled unless the caller explicitly enables it, and an injected tracker does
not change that selection rule. Leaf-regression epoch events are scalar-only:
the tracker receives computed scalar metrics, never their source prediction or
target tensors. The :doc:`leaf_regression` guide explains the CV and refit
stages; :doc:`../reference/leaf_regression` contains the three tracking-enabled
API signatures.

Common failures
---------------

`TrackingError` is raised for invalid tracking settings, missing wandb, unsafe
metadata, invalid metric payloads, and backend logging failures. Disable
tracking to keep the workflow local while debugging model or data issues. See
:doc:`../troubleshooting` for setup and failure guidance.

Related pages
-------------

See :doc:`training`, :doc:`training_config`, :doc:`../reference/training`, and
:doc:`../troubleshooting`.
