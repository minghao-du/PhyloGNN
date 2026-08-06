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
