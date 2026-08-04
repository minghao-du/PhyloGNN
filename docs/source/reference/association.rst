Association Reference
=====================

Public imports
--------------

.. code-block:: python

   from phylognn import (
       MaskedAttentionPhyloRegressor,
       RegionAssociationResult,
       build_leaf_laplacian,
       evaluate_region_association,
   )

Tensor contracts
----------------

``MaskedAttentionPhyloRegressor`` accepts a finite float32 normalized leaf
Laplacian of shape ``[N, N]``. Its forward method accepts finite float32
representations ``[N, L, D]`` and a bool-convertible position mask ``[N, L]``
with at least one valid position per row. It returns predictions ``[N]`` and
attention ``[N, L]``. Masked attention positions are exactly zero and each
valid row sums to one.

``evaluate_region_association`` uses the same representation and mask shapes.
Its target is a finite ``[N]`` tensor in leaf order or a mapping keyed by the
complete tree leaf-name set. It returns only a
``RegionAssociationResult``; it does not persist predictions or aggregate
regions.

Exceptions
----------

Unsupported tree or tensor object types raise ``TypeError``. Invalid tree leaf
names, alignment, shapes, dtypes, finite values, empty mask rows, folds,
undefined validation R-squared, and optimization settings raise ``ValueError``
before a misleading result is returned where the failure is determinable.

API
---

.. automodule:: phylognn.association
   :members: RegionAssociationResult, build_leaf_laplacian, evaluate_region_association
   :undoc-members:

.. autoclass:: phylognn.models.masked_attention.MaskedAttentionPhyloRegressor
   :members: forward

Related guide
-------------

See :doc:`../user_guide/region_association` for alignment, masking, normalized
leaf constraints, transductive CV, interpretation, and scope limits.
