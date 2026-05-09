Complete Pipeline
=================

This example maps to ``examples/complete_pipeline.py`` and demonstrates the
full local path from a tree to prediction. It uses the standard TOML training
checkpoint when present, and creates a temporary checkpoint internally when the
standard checkpoint has not been generated yet.

Inputs
------

- A tiny in-memory ``ete3.Tree`` created inside the script.
- Feature order ``("node_time", "time_bin", "branch_length", "is_tip")``.
- TOML model settings from ``examples/toml_training_config.toml``.
- Optional checkpoint ``example_outputs/toml_training_config/final_model.pt``.

Actions
-------

Run the pipeline directly from the repository root:

.. code-block:: bash

   python examples/complete_pipeline.py

The pipeline applies ``TreeFeatureEngineer.add_features()``, converts the tree
with ``TreeToGraphConverter``, creates a matching trainer from the TOML config,
loads the standard checkpoint when available, or creates a temporary checkpoint
for the same model before calling ``Trainer.predict()``.

Expected outputs
----------------

The script prints stable markers showing the checkpoint, graph tensor shape,
and prediction value:

.. code-block:: text

   Complete pipeline summary
   checkpoint:
   graph x shape:
   prediction:

Failure modes
-------------

Invalid graph fields fail through the existing model and trainer validation
paths. A missing standard checkpoint is handled internally with a temporary
checkpoint.

Optional settings
-----------------

The pipeline uses the same local, tracking-disabled TOML configuration as the
training example. Optional tracking and external services are outside this
default path.

Source
------

.. literalinclude:: ../../../examples/complete_pipeline.py
   :language: python
