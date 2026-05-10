Feature Engineering
===================

Script: ``examples/feature_engineering.py``.

Inputs
------

- A tiny in-memory ``ete3.Tree`` created inside the script.
- Feature order ``["node_time", "time_bin", "branch_length", "is_tip", "is_internal"]``.

Run command
-----------

Run the script from the repository root:

.. code-block:: bash

   python examples/feature_engineering.py

The script creates a ``TreeFeatureEngineer``, writes features onto each node,
and prints a compact node-by-node listing.

Expected output
---------------

Stable stdout markers include:

.. code-block:: text

   Feature engineering summary
   Feature order:
   root: node_time=

Files written
-------------

None.

Optional dependencies
---------------------

None.

Failure modes
-------------

Invalid feature names or tree inputs fail through the existing
``TreeFeatureEngineer`` validation paths.

Source
------

.. literalinclude:: ../../../examples/feature_engineering.py
   :language: python
