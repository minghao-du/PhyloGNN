Quickstart
==========

This first tutorial creates an `ete3.Tree`, attaches node features, converts it
to a PyTorch Geometric `Data` object, and inspects the main graph fields:
`data.x`, `data.edge_index`, `data.edge_type`, and `data.node_names`.

Create a small tree
-------------------

.. doctest::

   >>> from ete3 import Tree
   >>> tree = Tree("((A:1,B:1)C:1,D:2)Root:0;", format=1)
   >>> len(list(tree.traverse("preorder")))
   5

Attach node features
--------------------

`TreeFeatureEngineer` writes numeric attributes to each tree node. Use
`feature_names` as the stable column order for graph conversion.

.. doctest::

   >>> from phylognn import TreeFeatureEngineer
   >>> engineer = TreeFeatureEngineer(num_time_bins=4)
   >>> feature_names = ("node_time", "time_bin", "is_tip", "branch_length")
   >>> featured_tree = engineer.add_features(
   ...     tree,
   ...     origin_time=3.0,
   ...     feature_names=feature_names,
   ...     rescale=False,
   ...     inplace=False,
   ... )
   >>> all(hasattr(node, "time_bin") for node in featured_tree.traverse())
   True

Convert the tree to graph data
------------------------------

`TreeToGraphConverter` reads the node attributes into `data.x` and builds
tree edges in `data.edge_index`.

.. doctest::

   >>> from phylognn import TreeToGraphConverter
   >>> converter = TreeToGraphConverter(
   ...     feature_names=feature_names,
   ...     add_virtual_nodes=False,
   ...     append_is_virtual_feature=False,
   ... )
   >>> data = converter.convert(featured_tree)
   >>> tuple(data.x.shape)
   (5, 4)
   >>> tuple(data.edge_index.shape)
   (2, 8)
   >>> data.edge_type.tolist()
   [0, 0, 0, 0, 0, 0, 0, 0]
   >>> data.node_names
   ['Root', 'C', 'A', 'B', 'D']

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
