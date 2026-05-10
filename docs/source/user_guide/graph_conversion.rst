Graph Conversion
================

`TreeToGraphConverter` converts an `ete3.Tree` with numeric node attributes into
a PyTorch Geometric `Data` object.

Basic workflow
--------------

.. code-block:: python

   from phylognn import TreeToGraphConverter

   converter = TreeToGraphConverter(feature_names=engineer.feature_names)
   data = converter.convert(tree, graph_attrs={"tree_id": "example"})

The converter expects every node to have every requested feature and every
feature value to be numeric.

When to use it
--------------

Use this step after feature engineering and before any model or training code.
Keep `feature_names` explicit when comparing experiments so `data.x` columns
remain stable.

Output fields
-------------

`data.x`
   Floating-point node feature matrix with shape `[num_nodes, num_features]`.
   Column order is defined by `feature_names`, commonly
   `TreeFeatureEngineer.feature_names`.

`data.edge_index`
   `torch.long` tensor with shape `[2, num_edges]`. Tree parent-child
   relations are included, and bidirectional conversion adds reverse edges.

`data.edge_type`
   `torch.long` tensor aligned with `data.edge_index`. Values are `0` for tree
   edges, `1` for virtual-to-real edges, and `2` for virtual-chain edges.

`data.node_names`
   Optional list aligned with graph node order when `preserve_node_names=True`.
   Original nodes use ETE names, unnamed nodes use an empty string, and virtual
   nodes use generated names.

`data.original_num_nodes`
   Count of nodes from the original tree before virtual nodes are appended.

Converted data also includes `data.virtual_node_mask` and `data.node_type`.
User-provided `graph_attrs` are attached as graph-level attributes, except for
reserved generated field names such as `time_bin`.

When `feature_names` includes `time_bin`, the converter also attaches
`data.time_bin` as a one-dimensional `torch.long` tensor with one label per
final graph node. The labels follow the same row order as `data.x`. When
`feature_names` does not include `time_bin`, the converter does not infer or
attach `data.time_bin`.

Virtual nodes
-------------

Set `add_virtual_nodes=True` to add one virtual node per time bin. In this
mode, `feature_names` must include `time_bin`. Virtual-to-real edges have
`edge_type=1`; virtual-chain edges have `edge_type=2`. If
`append_is_virtual_feature=True`, the final feature column identifies virtual
nodes. If `num_time_bins` is configured, one virtual node is created for every
configured bin, including empty bins. Generated `data.time_bin` labels for
virtual nodes are appended in ascending bin order.

Deterministic ordering
----------------------

Feature order is deterministic when callers pass an ordered sequence such as
`TreeFeatureEngineer.feature_names`. Node order follows the converter
traversal strategy, with `preorder` as the default. Metadata aligned to nodes,
including `node_names`, follows that same order.

Common validation errors
------------------------

The converter raises clear errors for missing requested node attributes,
non-numeric feature values, duplicate feature names, unsupported traversal
strategies, invalid virtual-node settings, and graph attribute names that
collide with generated fields.

Saving and loading
------------------

Use `convert_and_save()` for preprocessing pipelines, `save_data()` to store a
PyTorch Geometric `Data` object, and `load_data()` to restore it with
`torch.load`. Saved graph files are complete PyTorch objects; load them only
from trusted PhyloGNN project outputs.

Related pages
-------------

See :doc:`feature_engineering` for node features, :doc:`training` for model
input expectations, and :doc:`../reference/data` for the public API.
