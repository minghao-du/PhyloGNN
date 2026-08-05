Leaf Regression
===============

The leaf regression workflow predicts one continuous target for each leaf of an
in-memory ``ete3.Tree``. It is independent of the package's PyTorch Geometric
graph conversion and trainer workflows.

Leaf alignment
--------------

``representations`` has shape ``[N, L, D]``: one padded sequence of position
vectors for each tree leaf. ``position_mask`` has shape ``[N, L]`` and marks
valid positions. It accepts booleans or finite numeric ``0`` and ``1`` values,
then becomes a boolean mask; every row must contain at least one valid position.

The default row order is ``tree.iter_leaves()``. Supplying ``leaf_names`` is
allowed only when it is a complete, unique permutation of those names. Positional
targets must already be in that final order. Mapping targets must have exactly
the leaf-name keys and are aligned by the final leaf order. Preparation never
reorders caller representation or mask rows.

Recommended workflow
--------------------

Use :func:`phylognn.leaf_regression.run_leaf_regression` for preparation,
transductive cross-validation, and one final all-leaf refit:

.. code-block:: python

   from phylognn import LeafRegressionConfig, run_leaf_regression

   result = run_leaf_regression(
       tree,
       representations,
       position_mask,
       targets,
       n_splits=3,
       training_config=LeafRegressionConfig(epochs=100, seed=7),
   )

``result.oof_predictions`` contains one held-out prediction per leaf.
``result.predictions`` comes from the final refit. ``cv_score`` is the
validation-leaf-count-weighted mean of the fold scores.

Staged workflow and scoring
---------------------------

For explicit control, call :func:`phylognn.leaf_regression.prepare_leaf_regression`,
:func:`phylognn.leaf_regression.fit_leaf_regression`, or
:func:`phylognn.leaf_regression.cross_validate_leaf_regression`. Manual
validation folds retain their input order, must be disjoint, and must cover
every leaf exactly once.
Without a custom score, every validation fold uses R-squared and therefore
needs at least two leaves with nonconstant targets.

Each fold sees the full tree constraint and all representations, but its loss
uses only the training-leaf targets. This is transductive cross-validation: it
measures held-out targets when the complete tree and leaf representations are
available, rather than predicting an unseen leaf.

Model construction and attention
--------------------------------

By default, the workflow creates a fresh ``MaskedAttentionPhyloRegressor`` for
each fold and final refit. It injects the prepared representation width and leaf
constraint. A
custom model must be an ``torch.nn.Module`` subclass and is constructed from
the supplied keyword mapping for every fit.

Models may return predictions alone or a prediction-and-attention pair. When
attention is present, it has shape ``[N, L]`` with masked positions exactly
zero and ``mean_attention`` has shape ``[L]``. Prediction-only models return
``None`` for both attention fields; the workflow does not synthesize an
interpretation value.

Errors and determinism
----------------------

Malformed trees, tensors, masks, targets, configurations, folds, scoring
results, model construction, model outputs, and first losses fail with clear
``TypeError`` or ``ValueError`` exceptions before the first optimizer update.
With a fixed ``LeafRegressionConfig.seed``, leaf order, generated folds, stage
initializations, predictions, losses, and scores are reproducible where the
selected PyTorch operations support deterministic execution. The workflow
restores the caller's Python, NumPy, and PyTorch random-number-generator state
after successful and exceptional calls.

The workflow does not perform multi-tree aggregation, significance testing,
target transformation, missing-value handling, causal inference, persistence,
checkpoints, plots, or file I/O. Attention describes model weighting of input
positions; it is not evidence of a causal biological effect.

See :doc:`../reference/leaf_regression` for the complete API and
:doc:`../examples/single_tree_leaf_regression` for a runnable in-memory example.
