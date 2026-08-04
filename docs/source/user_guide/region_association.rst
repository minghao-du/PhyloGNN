Single-Tree Region Association
==============================

The region association workflow evaluates exactly one leaf-aligned region on
one in-memory ``ete3.Tree``. It is independent of the package's PyTorch
Geometric graph conversion and trainer workflows.

Leaf alignment
--------------

``representations`` has shape ``[N, L, D]``: one sequence of position vectors
for every tree leaf. ``position_mask`` has shape ``[N, L]`` and marks valid
positions, allowing each leaf to have a different sequence length while
preserving a padded tensor. Every mask row must contain a valid position.

The default row order is ``tree.iter_leaves()``. Supplying ``leaf_names`` is
allowed only when it is a complete, unique permutation of the tree leaf names.
Targets are either an already aligned ``[N]`` tensor or a mapping keyed by
exactly those leaf names. The evaluator never sorts, filters, pads, or imputes
leaf data.

Tree constraint and model
-------------------------

``build_leaf_laplacian`` constructs a finite float32 normalized Laplacian over
the leaves from pairwise tree distances. For a zero-length tree it uses topology
distance so the leaf constraint remains meaningful. The masked-attention model
assigns exactly zero attention to padding, normalizes attention across each
leaf's valid positions, and learns a sigmoid-bounded smoothing strength over
the normalized leaf constraint.

Transductive cross-validation
-----------------------------

``evaluate_region_association`` creates deterministic shuffled folds from its
``seed``. In every fold the forward pass sees all leaf representations and the
complete tree constraint, while loss uses only the training targets and
R-squared uses only validation targets. A fresh model is then refit on all
targets to produce attention. This is transductive CV, so its score measures
held-out targets with the full tree and representations available, not a
prediction for unseen leaves.

Results and limits
------------------

The returned :class:`phylognn.association.RegionAssociationResult` contains ``fold_r2``,
their arithmetic mean ``cv_r2``, per-leaf ``attention`` with shape ``[N, L]``,
and ``mean_attention`` with shape ``[L]``. Result fields cannot be reassigned;
the returned tensors are detached clones.

This workflow does not perform multi-tree or multi-region aggregation,
significance testing, target transformation, missing-value handling, causal
inference, persistence, checkpoints, plots, or file I/O. Attention highlights
model weighting of input positions; it is not evidence of a causal biological
effect.

See :doc:`../reference/association` for signatures and
:doc:`../examples/single_tree_region_association` for a runnable in-memory
example.
