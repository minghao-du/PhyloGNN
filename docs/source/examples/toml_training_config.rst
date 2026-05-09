TOML Training Configuration
===========================

This example maps ``examples/toml_training_config.toml`` to a complete local
training run in ``examples/toml_training_config.py``. It demonstrates how a
TOML file configures the model, trainer, loss, metrics, and tracking boundary
while data still comes from ordinary Python code.

Inputs
------

- ``examples/toml_training_config.toml`` with ``[model]``, ``[model.params]``,
  ``[training]``, ``[loss]``, ``[metrics]``, and ``[tracking]`` sections.
- A deterministic in-memory graph dataset created by the script from small
  ``ete3.Tree`` objects.
- Feature order ``("node_time", "time_bin", "branch_length", "is_tip")``.

Actions
-------

Run the script from the repository root:

.. code-block:: bash

   python examples/toml_training_config.py

The script loads the TOML config through ``load_training_config()``, creates a
``Trainer`` through ``create_trainer_from_config()``, builds train and
validation splits, and calls ``Trainer.fit()``.

Expected outputs
----------------

The script prints stable markers for smoke tests and writes the checkpoint and
history files used by the complete pipeline:

.. code-block:: text

   TOML training run summary
   configured model: GATBiLSTMNet
   checkpoint: example_outputs/toml_training_config/final_model.pt
   history: example_outputs/toml_training_config/history.json

Files created:

- ``example_outputs/toml_training_config/final_model.pt``
- ``example_outputs/toml_training_config/history.json``

Failure modes
-------------

- Missing or malformed TOML raises ``TrainingConfigError`` from the training
  configuration loader.
- Invalid model or trainer keys fail during configuration validation before
  training starts.
- Output files are regenerated on each run, so stale files can be removed by
  deleting ``example_outputs/toml_training_config/``.

Optional settings
-----------------

The default ``[tracking]`` section keeps experiment tracking disabled. Install
and configure the ``wandb`` extra only when you intentionally enable tracking;
this local example does not require credentials.

