Association Reference
=====================

Public imports
--------------

.. code-block:: python

   from phylognn import (
       MaskedAttentionPhyloRegressor,
       RegionAssociationData,
       RegionAssociationCVResult,
       RegionFitConfig,
       RegionFitResult,
       RegionAssociationResult,
       build_leaf_laplacian,
       cross_validate_region_association,
       evaluate_region_association,
       fit_region_association,
       prepare_region_association,
   )

Tensor contracts
----------------

``MaskedAttentionPhyloRegressor`` accepts a finite float32 normalized leaf
Laplacian of shape ``[N, N]``. Its forward method accepts finite float32
representations ``[N, L, D]`` and a bool-convertible position mask ``[N, L]``
with at least one valid position per row. It returns predictions ``[N]`` and
attention ``[N, L]``. Masked attention positions are exactly zero and each
valid row sums to one.

``prepare_region_association`` returns reusable frozen data with those tensors,
ordered leaf names, and a finite ``[N, N]`` leaf constraint. ``fit_region_association``
accepts that object and returns detached all-leaf predictions and masked
attention while training on all or selected leaves. ``cross_validate_region_association``
preserves or generates complete validation folds, returns one OOF prediction per
leaf, and optionally exposes one all-leaf ``final_fit``. None of these APIs
persist predictions or aggregate regions.

``evaluate_region_association`` remains the compatible one-shot entry point. Its
target is a finite ``[N]`` tensor in leaf order or a mapping keyed by the
complete tree leaf-name set, and it returns the legacy
``RegionAssociationResult`` fields derived from the staged result.

Exceptions
----------

Unsupported tree or tensor object types raise ``TypeError``. Invalid tree leaf
names, alignment, shapes, dtypes, finite values, empty mask rows, folds,
undefined validation R-squared, and optimization settings raise ``ValueError``
before a misleading result is returned where the failure is determinable.

API
---

.. automodule:: phylognn.association
   :members: RegionAssociationData, RegionFitConfig, RegionFitResult, RegionAssociationCVResult, RegionAssociationResult, build_leaf_laplacian, prepare_region_association, fit_region_association, cross_validate_region_association, evaluate_region_association
   :undoc-members:

.. autoclass:: phylognn.models.masked_attention.MaskedAttentionPhyloRegressor
   :members: forward

Related guide
-------------

See :doc:`../user_guide/region_association` for alignment, masking, normalized
leaf constraints, transductive CV, interpretation, and scope limits.
