TOML Training Configuration
===========================

Script: ``examples/toml_training_config.py``.
Configuration: ``examples/toml_training_config.toml``.

Inputs
------

- ``examples/toml_training_config.toml`` with ``[model]``, ``[model.params]``,
  ``[training]``, ``[loss]``, ``[metrics]``, and ``[tracking]`` sections.
- A deterministic in-memory graph dataset created by the script from small
  ``ete3.Tree`` objects.
- Feature order ``("node_time", "time_bin", "branch_length", "is_tip")``.

Run command
-----------

Run the script from the repository root:

.. code-block:: bash

   python examples/toml_training_config.py

The script creates a ``Trainer`` through ``create_trainer_from_config()``,
builds train and validation splits, and calls ``Trainer.fit()``.

Expected output
---------------

The script prints stable markers for smoke tests and writes the checkpoint and
history files used by the complete pipeline:

.. code-block:: text

   TOML training run summary
   configured model: GATBiLSTMNet
   checkpoint: example_outputs/toml_training_config/final_model.pt
   history: example_outputs/toml_training_config/history.json

Stable stdout markers include:

.. code-block:: text

   TOML training run summary
   configured model: GATBiLSTMNet
   metrics: mse, rmse
   checkpoint: example_outputs/toml_training_config/final_model.pt
   history: example_outputs/toml_training_config/history.json

Files written
-------------

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

Optional dependencies
---------------------

The default ``[tracking]`` section keeps experiment tracking disabled. Install
and configure the ``wandb`` extra only when you intentionally enable tracking;
this local example does not require credentials.

Source
------

.. literalinclude:: ../../../examples/toml_training_config.py
   :language: python

.. literalinclude:: ../../../examples/toml_training_config.toml
   :language: toml
