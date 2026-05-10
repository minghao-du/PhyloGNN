Tree To Graph
=============

Script: ``examples/tree_to_graph.py``.

Inputs
------

- A tiny in-memory ``ete3.Tree`` created inside the script.
- Feature order ``("node_time", "time_bin", "branch_length", "is_tip")``.
- A virtual-node conversion variant using the same featured tree.

Run command
-----------

Run the script from the repository root:

.. code-block:: bash

   python examples/tree_to_graph.py

The script applies ``TreeFeatureEngineer.add_features()``, converts the tree
with ``TreeToGraphConverter``, then repeats conversion with virtual time-bin
nodes enabled.

Expected output
---------------

Stable stdout markers include:

.. code-block:: text

   Graph summary
   x shape:
   edge_index shape:
   num_nodes:
   virtual node count:

Files written
-------------

None.

Optional dependencies
---------------------

None.

Failure modes
-------------

Invalid feature names, missing required node attributes, or incompatible
virtual-node settings fail through the existing converter validation paths.

Source
------

.. literalinclude:: ../../../examples/tree_to_graph.py
   :language: python
