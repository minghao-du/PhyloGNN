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

The staged API makes those decisions explicit. First,
``prepare_region_association`` validates and freezes the leaf order, tensors,
mask, targets, and leaf constraint for reuse. Then
``cross_validate_region_association`` creates deterministic shuffled folds from
``RegionFitConfig.seed`` (or preserves supplied folds), calls
``fit_region_association`` for each training complement, assembles one OOF
prediction per leaf, and optionally performs one all-leaf refit. In every fold
the forward pass sees all leaf representations and the complete tree
constraint, while loss uses only the training targets and R-squared uses only
validation targets.

The original ``evaluate_region_association`` function remains the short path.
It builds ``RegionFitConfig`` from its existing keywords and delegates to
preparation plus cross-validation with ``refit=True``; its attention comes
directly from the returned final fit, without a duplicate full-data fit. This
is transductive CV, so its score measures held-out targets with the full tree
and representations available, not a prediction for unseen leaves.

Results and limits
------------------

The staged :class:`phylognn.association.RegionAssociationData` and
:class:`phylognn.association.RegionFitConfig` objects are frozen contracts.
``RegionAssociationCVResult`` contains ``fold_scores``, ``cv_score``, complete
``oof_predictions``, the ordered ``validation_folds``, one ``fold_results``
entry per fold, and an optional ``final_fit``. Every fit result contains
detached predictions, zero-padded attention, selected ``train_indices``, and
finite per-epoch losses. The compatibility
:class:`phylognn.association.RegionAssociationResult` contains the same scores
under its legacy names plus the final-fit attention. Dataclass fields cannot
be reassigned; returned tensors are detached clones.

This workflow does not perform multi-tree or multi-region aggregation,
significance testing, target transformation, missing-value handling, causal
inference, persistence, checkpoints, plots, or file I/O. Attention highlights
model weighting of input positions; it is not evidence of a causal biological
effect.

See :doc:`../reference/association` for signatures and
:doc:`../examples/single_tree_region_association` for a runnable in-memory
example.
