Leaf Regression Reference
=========================

Public imports
--------------

The root package and :mod:`phylognn.leaf_regression` expose exactly these Leaf
Regression names:

.. code-block:: python

   from phylognn import (
       LeafRegressionData,
       LeafRegressionConfig,
       LeafFitResult,
       LeafCrossValidationResult,
       LeafRegressionResult,
       prepare_leaf_regression,
       fit_leaf_regression,
       cross_validate_leaf_regression,
       run_leaf_regression,
   )

Tensor and model contracts
--------------------------

Preparation returns frozen leaf-aligned data: finite float32 representations
``[N, L, D]``, a boolean mask ``[N, L]``, finite float32 targets ``[N]``, and a
finite float32 internal leaf constraint ``[N, N]``. Fit results provide
all-leaf predictions and optional attention while training on all or selected
leaves. Cross-validation returns complete OOF predictions, ordered folds,
weighted scores, fold results, and an optional final fit. Validation is
transductive: every fold uses the complete tree constraint and leaf
representations while holding out only target values from its training loss.

The default model is ``MaskedAttentionPhyloRegressor``. It consumes the
prepared leaf constraint, returns predictions ``[N]`` and masked attention
``[N, L]``, and assigns exactly zero attention to padding.
Custom models may instead return predictions alone.

API
---

.. automodule:: phylognn.leaf_regression
   :members: LeafRegressionData, LeafRegressionConfig, LeafFitResult, LeafCrossValidationResult, LeafRegressionResult, prepare_leaf_regression, fit_leaf_regression, cross_validate_leaf_regression, run_leaf_regression
   :undoc-members:

Related guide
-------------

See :doc:`../user_guide/leaf_regression` for input alignment, staged usage,
scoring, attention, errors, determinism, and workflow limits.
