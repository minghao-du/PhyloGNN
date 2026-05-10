Feature Engineering
===================

`TreeFeatureEngineer` computes numeric node attributes on an `ete3.Tree`. The
converter later reads those attributes into graph feature columns.

Basic workflow
--------------

.. code-block:: python

   from phylognn import TreeFeatureEngineer

   engineer = TreeFeatureEngineer(num_time_bins=101)
   tree = engineer.add_features(tree, origin_time=10.0, rescale=True)
   feature_order = engineer.feature_names

Built-in features include `node_time`, `time_bin`, tip/internal indicators,
fossil/extant indicators, sampled-ancestor indicators, `branch_length`,
`rescale_factor`, and `extant_sampling_probability`.

When to use it
--------------

Run feature engineering before graph conversion whenever the converter should
read computed node attributes into graph feature columns. Keep custom features
numeric and registered by name before requesting them.

Feature order and determinism
-----------------------------

`feature_names` is an immutable ordered tuple. Use it when constructing a
converter so feature columns stay stable across runs. Custom features are
appended after built-in features in registration order.

Validation
----------

`origin_time` must be positive. Requested feature names must be unique and
must exist in `available_features`. `num_time_bins` must be at least two,
`extant_sampling_probability` must be in `[0, 1]`, and traversal strategy must
be one of `preorder`, `postorder`, or `levelorder`. These contracts are checked
before graph conversion so invalid features fail early.

Rescaling
---------

When `rescale=True`, non-zero branch lengths are scaled so their mean becomes
one. The same factor is used for feature computation and is attached as
`rescale_factor`. Trees with no non-zero branch lengths cannot be rescaled.

Related pages
-------------

See :doc:`graph_conversion` for converting features to graph tensors,
:doc:`../reference/data` for API details, and :doc:`../examples/feature_engineering`
for the runnable script.
