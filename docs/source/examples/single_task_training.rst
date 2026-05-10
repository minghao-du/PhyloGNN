Single-Task Training
====================

Script: ``examples/single_task_training.py``.

Inputs
------

- A tiny deterministic in-memory graph dataset created by the script.
- Feature order ``["node_time", "time_bin", "branch_length", "is_tip"]``.
- A minimal graph regressor defined inside the example.

Run command
-----------

Run the script from the repository root:

.. code-block:: bash

   python examples/single_task_training.py

The script creates train, validation, and test splits, trains for a few epochs,
and runs prediction on the test split.

Expected output
---------------

Stable stdout markers include:

.. code-block:: text

   Training summary
   dataset sizes:
   final losses:
   output_dir:
   prediction sample:

Files written
-------------

Temporary checkpoints and history are written under a temporary directory and
removed when the script exits.

Optional dependencies
---------------------

None.

Failure modes
-------------

Invalid graph fields or trainer settings fail through the existing model and
trainer validation paths.

Source
------

.. literalinclude:: ../../../examples/single_task_training.py
   :language: python
