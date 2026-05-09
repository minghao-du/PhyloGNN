Single-Task Training
====================

This example maps to ``examples/single_task_training.py`` and demonstrates a
compact end-to-end single-task training workflow.

Inputs
------

- A tiny deterministic in-memory graph dataset created by the script.
- Feature order ``["node_time", "time_bin", "branch_length", "is_tip"]``.
- A minimal graph regressor defined inside the example.

Actions
-------

Run the script from the repository root:

.. code-block:: bash

   python examples/single_task_training.py

The script creates train, validation, and test splits, trains for a few epochs,
and runs prediction on the test split.

Expected outputs
----------------

The script prints a ``Training summary``, split sizes, final losses, a
temporary output directory, and one prediction sample. Training artifacts are
written under a temporary directory that is removed when the script exits.

Failure modes
-------------

Invalid graph fields or trainer settings fail through the existing model and
trainer validation paths.

Optional settings
-----------------

This local example keeps external tracking disabled and does not require
optional service credentials.

Source
------

.. literalinclude:: ../../../examples/single_task_training.py
   :language: python
