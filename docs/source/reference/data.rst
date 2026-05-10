Data Reference
==============

Import path
-----------

.. code-block:: python

   from phylognn.data import TreeFeatureEngineer, TreeToGraphConverter

Feature engineering
-------------------

.. py:class:: TreeFeatureEngineer(num_time_bins=101, extant_sampling_probability=1.0, custom_features=None, traversal_strategy="preorder", time_tolerance=1e-8)

   Attach numeric node-level features to an `ete3.Tree`.

   Important attributes are `feature_names`, an ordered immutable feature list,
   and `available_features`, an immutable membership set. Use
   `feature_names` as the column order for graph conversion.

   Built-in features include `node_time`, `time_bin`, `is_internal`, `is_tip`,
   `is_fossil`, `is_extant`, sampled-ancestor indicators, `branch_length`,
   `rescale_factor`, and `extant_sampling_probability`.

   Common failures include invalid constructor values, non-positive
   `origin_time`, duplicate feature requests, unknown feature names, and
   rescaling trees with no non-zero branch lengths.

   .. py:method:: add_features(tree, origin_time, feature_names=None, rescale=True, inplace=True)

      Compute requested node features and attach them to each node.

   .. py:method:: rescale_tree(tree, inplace=True)

      Rescale non-zero branch lengths so their mean becomes one.

   .. py:method:: get_available_features()

      Return the registered feature names in stable order.

Graph conversion
----------------

.. py:class:: TreeToGraphConverter(feature_names=None, add_virtual_nodes=False, num_time_bins=None, traversal_strategy="preorder", bidirectional=True, connect_virtual_to_real=True, connect_virtual_chain=True, append_is_virtual_feature=True, preserve_node_names=True, copy_sampling_prob_to_virtual=True)

   Convert a feature-bearing `ete3.Tree` to a PyTorch Geometric `Data` object.

   `feature_names` defines the feature column order. Every node must contain
   every requested feature and each feature value must be numeric.

   Output fields include `x`, `edge_index`, `edge_type`, `original_num_nodes`,
   `virtual_node_mask`, `node_type`, optional `node_names`, optional
   `num_time_bins`, and user-provided graph-level attributes.

   Edge type values are `0` for tree edges, `1` for virtual-to-real edges, and
   `2` for virtual-chain edges.

   .. py:method:: convert(tree, graph_attrs=None)

      Return graph data for one tree.

   .. py:method:: convert_and_save(tree, path, graph_attrs=None, create_dirs=True)

      Convert a tree, save the graph, and return it.

   .. py:method:: save_data(data, path, create_dirs=True)

      Save a PyG `Data` object with `torch.save`.

   .. py:staticmethod:: load_data(path, map_location=None)

      Load a saved PyG `Data` object.

Related guide
-------------

See :doc:`../user_guide/feature_engineering` and
:doc:`../user_guide/graph_conversion`.
