Complete Pipeline
=================

Script: ``examples/complete_pipeline.py``.

Inputs
------

- A tiny in-memory ``ete3.Tree`` created inside the script.
- Feature order ``("node_time", "time_bin", "branch_length", "is_tip")``.
- TOML model settings from ``examples/toml_training_config.toml``.
- Optional checkpoint ``example_outputs/toml_training_config/final_model.pt``.

Run command
-----------

Run the pipeline directly from the repository root:

.. code-block:: bash

   python examples/complete_pipeline.py

The pipeline applies ``TreeFeatureEngineer.add_features()``, converts the tree
with ``TreeToGraphConverter``, creates a matching trainer from the TOML config,
loads the standard checkpoint when available, or creates a temporary checkpoint
for the same model before calling ``Trainer.predict()``.

Expected output
---------------

Stable stdout markers include:

.. code-block:: text

   Complete pipeline summary
   checkpoint:
   graph x shape:
   prediction:

Files written
-------------

If ``example_outputs/toml_training_config/final_model.pt`` exists, the script
loads it. If it is missing, the script creates a temporary checkpoint
internally and removes it before exit.

Optional dependencies
---------------------

None.

Failure modes
-------------

Invalid graph fields fail through the existing model and trainer validation
paths. A missing standard checkpoint is handled internally with a temporary
checkpoint.

Source
------

.. literalinclude:: ../../../examples/complete_pipeline.py
   :language: python
