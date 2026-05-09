Complete Pipeline
=================

This example maps to ``examples/complete_pipeline.py`` and demonstrates the
full local path from a tree to prediction. It consumes the checkpoint produced
by :doc:`toml_training_config`.

Inputs
------

- A tiny in-memory ``ete3.Tree`` created inside the script.
- Feature order ``("node_time", "time_bin", "branch_length", "is_tip")``.
- TOML model settings from ``examples/toml_training_config.toml``.
- Checkpoint ``example_outputs/toml_training_config/final_model.pt``.

Actions
-------

Run the training example first, then run the pipeline:

.. code-block:: bash

   python examples/toml_training_config.py
   python examples/complete_pipeline.py

The pipeline applies ``TreeFeatureEngineer.add_features()``, converts the tree
with ``TreeToGraphConverter``, creates a matching trainer from the TOML config,
loads ``final_model.pt``, and calls ``Trainer.predict()``.

Expected outputs
----------------

The script prints stable markers showing the checkpoint, graph tensor shape,
and prediction value:

.. code-block:: text

   Complete pipeline summary
   checkpoint: example_outputs/toml_training_config/final_model.pt
   graph x shape:
   prediction:

Failure modes
-------------

If the checkpoint is missing, the script exits with a clear message telling the
user to run ``python examples/toml_training_config.py`` first. Invalid graph
fields fail through the existing model and trainer validation paths.

Optional settings
-----------------

The pipeline uses the same local, tracking-disabled TOML configuration as the
training example. Optional tracking and external services are outside this
default path.
