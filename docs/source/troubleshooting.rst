Troubleshooting
===============

Use this page to map common failures to the workflow area that owns the fix.
Source code and tests remain authoritative when an error message and a guide
appear to disagree.

Malformed or unsupported tree input
-----------------------------------

If `ETE3` cannot parse a `Newick` string or `DendroPy` cannot parse a file, first
confirm the tree format and schema. Core workflows accept `ete3.Tree` objects.
`NEXUS` or `BEAST`-style file reading through `DendroPy` requires the `beast` or
`all` extra.

Missing optional dependencies
-----------------------------

`ModuleNotFoundError` for `DendroPy` means optional tree I/O was used without the
`beast` extra. `TrackingError` mentioning wandb means tracking was enabled
without the `wandb` extra.

Leaf-regression tracking is disabled by default. Install the optional backend
only when an external run is needed:

.. code-block:: bash

   python -m pip install -e ".[wandb]"

The default leaf-regression example and all three public entry points remain
offline and file-free without this extra. Supplying an injected tracker does
not change that behavior; pass ``TrackingConfig(enabled=True, project="...")``
explicitly to opt in.

Duplicate or unknown features
-----------------------------

`TreeFeatureEngineer.add_features()` rejects duplicate requested feature names
and unknown features. Use `engineer.feature_names` or
`engineer.get_available_features()` when you need the current supported order.

Invalid graph data
------------------

Converters require every requested feature to exist on every node and to be
numeric. See :doc:`user_guide/graph_conversion` for the canonical graph field
contract. Batched model training also needs `batch`, and temporal modes need
`data.time_bin`.

Invalid TOML training configuration
-----------------------------------

`TrainingConfigError` reports malformed `TOML`, missing sections, unknown keys,
wrong types, unsupported loss or metric names, and invalid trainer values. See
:doc:`user_guide/training_config` for the accepted sections.

Tracking setup failures
-----------------------

When tracking is enabled, `tracking.project` is required. Metadata keys that
look secret, such as tokens or passwords, are rejected before external logging.
Disable tracking to keep training fully local.

Only sanitized configuration, finite scalar metrics, stage identifiers, run
identity, and terminal status are sent. Trees, leaf names or index lists, input
and output tensors, attention, model state, checkpoints, and artifacts are not
uploaded. Path-like metadata is reduced to its final component; secret-like
metadata keys are rejected.

Metric-selection validation failures
------------------------------------

For enabled tracking, metric selection is validated before tracker start. Use
``TrackingConfig(metrics=None)`` for all applicable quantitative metrics,
``TrackingConfig(metrics=())`` for none, or an ordered tuple of catalog names
for an allowlist. In TOML, use ``[tracking].metrics = ["train/loss"]`` or
``metrics = []`` inside the ``[tracking]`` table.

Duplicate names, empty or malformed names, unknown fixed names, and dynamic
``train/<name>`` or ``val/<name>`` names not configured on the active trainer
fail before tracker start. Secret-like namespace segments also fail validation;
remove names containing ``api_key``, ``apikey``, ``auth``, ``credential``,
``password``, ``secret``, or ``token``. A catalog name that is valid but not
applicable to the current workflow is omitted from that event.

Missing or undefined metrics
----------------------------

Only finite scalar metrics are uploaded. An unavailable validation metric or a
metric excluded by selection is omitted, not replaced with zero. Pearson
correlation requires at least two aligned out-of-fold values and nonzero
variance in both inputs. When it is undefined, leaf regression raises a
``RuntimeWarning`` and omits ``cv/pearson_r`` while retaining every other
finite summary metric.

Lazy W&B loading and privacy
----------------------------

W&B is imported lazily only after enabled tracking begins. Leave tracking
disabled to avoid importing the optional dependency, starting a run, or
validating a workflow metric selection. Sanitized configuration, finite scalar
metrics, stage identifiers, and terminal state may be sent; raw trees, leaf
names, tensors, predictions, attention, model state, checkpoints, artifacts,
and full local paths are not uploaded.

Common tracking failures
~~~~~~~~~~~~~~~~~~~~~~~~

* ``tracking.project is required``: enable tracking with a non-empty W&B project.
* ``wandb is required``: install the ``wandb`` extra in the active environment.
* Unsafe metadata or unsupported values: remove token/password fields and keep
  metadata scalar or a flat scalar sequence.
* Backend initialization or logging errors: verify credentials and network
  access, or disable tracking while diagnosing the scientific workflow.
  Training failures remain the primary exception; cleanup failures after a
  failed or interrupted run are warnings.

Documentation build failures
----------------------------

Install the docs extra, then run the HTML and doctest builders from the
repository root:

.. code-block:: bash

   python -m pip install -e ".[docs]"
   python -m sphinx -b html -n -W --keep-going docs/source docs/_build/html
   python -m sphinx -b doctest -W docs/source docs/_build/doctest

Warnings are treated as failures so missing pages, broken references, and stale
quickstart snippets are visible before release.
