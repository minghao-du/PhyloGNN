Tree To Graph
=============

This example maps to ``examples/tree_to_graph.py`` and demonstrates converting
a featured tree into a PyTorch Geometric ``Data`` object.

Inputs
------

- A tiny in-memory ``ete3.Tree`` created inside the script.
- Feature order ``("node_time", "time_bin", "branch_length", "is_tip")``.
- A virtual-node conversion variant using the same featured tree.

Actions
-------

Run the script from the repository root:

.. code-block:: bash

   python examples/tree_to_graph.py

The script applies ``TreeFeatureEngineer.add_features()``, converts the tree
with ``TreeToGraphConverter``, then repeats conversion with virtual time-bin
nodes enabled.

Expected outputs
----------------

The script prints a ``Graph summary`` with tensor shapes, metadata, and virtual
node counts. It does not write output files.

Failure modes
-------------

Invalid feature names, missing required node attributes, or incompatible
virtual-node settings fail through the existing converter validation paths.

Optional settings
-----------------

Virtual nodes are demonstrated in the script. File-loading and tracking
dependencies are not required.

Source
------

.. literalinclude:: ../../../examples/tree_to_graph.py
   :language: python
