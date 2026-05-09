Feature Engineering
===================

This example maps to ``examples/feature_engineering.py`` and demonstrates
attaching deterministic numeric node features to an in-memory ``ete3.Tree``.

Inputs
------

- A tiny in-memory ``ete3.Tree`` created inside the script.
- Feature order ``["node_time", "time_bin", "branch_length", "is_tip", "is_internal"]``.

Actions
-------

Run the script from the repository root:

.. code-block:: bash

   python examples/feature_engineering.py

The script creates a ``TreeFeatureEngineer``, writes features onto each node,
and prints a compact node-by-node listing.

Expected outputs
----------------

The script prints a ``Feature engineering summary`` and the feature values for
each traversed node. It does not write output files.

Failure modes
-------------

Invalid feature names or tree inputs fail through the existing
``TreeFeatureEngineer`` validation paths.

Optional settings
-----------------

This example is self-contained and does not require optional file-loading or
tracking dependencies.

Source
------

.. literalinclude:: ../../../examples/feature_engineering.py
   :language: python
