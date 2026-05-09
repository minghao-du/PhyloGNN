Quickstart
==========

This first tutorial creates an `ete3.Tree`, attaches node features, converts it
to a PyTorch Geometric `Data` object, and inspects the main graph fields:
`data.x`, `data.edge_index`, `data.edge_type`, and `data.node_names`.

Create a small tree
-------------------

.. literalinclude:: ../../examples/tree_to_graph.py
   :language: python
   :start-after: [START build_demo_tree]
   :end-before: [END build_demo_tree]

Attach node features
--------------------

`TreeFeatureEngineer` writes numeric attributes to each tree node. Use
`feature_names` as the stable column order for graph conversion.

.. literalinclude:: ../../examples/tree_to_graph.py
   :language: python
   :start-after: [START feature_engineering]
   :end-before: [END feature_engineering]

Convert the tree to graph data
------------------------------

`TreeToGraphConverter` reads the node attributes into `data.x` and builds
tree edges in `data.edge_index`.

.. literalinclude:: ../../examples/tree_to_graph.py
   :language: python
   :start-after: [START tree_to_graph_conversion]
   :end-before: [END tree_to_graph_conversion]

Interpret the output
--------------------

`data.x` is a floating-point matrix with one row per graph node and one column
per requested feature. `data.edge_index` is a `LongTensor` with shape
`[2, num_edges]`; by default tree edges are bidirectional. `data.edge_type`
uses `0` for tree edges. `data.node_names` follows the converter traversal
order, and `data.original_num_nodes` records how many nodes came from the
input tree before any virtual nodes are added.

For virtual-node graphs, include `time_bin` in the feature order and enable
`add_virtual_nodes=True`. See :doc:`concepts/graph_data` and
:doc:`user_guide/graph_conversion` for the complete field contract.

Next steps
----------

Read :doc:`concepts/graph_data` for field semantics, then move to
:doc:`user_guide/index` for workflow pages that cover file input, feature
engineering, conversion, training, and tracking.
