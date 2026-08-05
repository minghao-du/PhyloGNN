Single-Tree Region Association
==============================

Script: ``examples/single_tree_region_association.py``.

Inputs
------

- A six-leaf in-memory ``ete3.Tree`` created in the script.
- Finite float32 leaf representations with shape ``[6, 4, 3]``.
- A ``[6, 4]`` position mask for variable-length leaf sequences and a finite
  continuous target vector with shape ``[6]``.

Run command
-----------

Run the script from the repository root after installing the core package:

.. code-block:: bash

   python examples/single_tree_region_association.py

Expected output
---------------

The small one-shot and staged transductive evaluations complete before the test
suite's 30-second hang guard and print stable markers:

.. code-block:: text

   Single-tree region association summary
   leaf count:
   representations shape:
   position mask shape:
   fold R2:
   cv R2:
   maximum mean-attention position:
   staged fold count:
   staged OOF predictions shape:
   staged final attention shape:

The displayed scores are a deterministic demonstration, not a biological
claim. The maximum mean-attention position identifies the largest averaged
valid-position attention weight.

Files written
-------------

None. The tree, tensors, models, folds, scores, and attention remain in memory.

Optional dependencies
---------------------

None. The example needs only the core package dependencies, including PyTorch
and ETE3.

Failure modes
-------------

The staged preparation and fit/CV boundaries reject malformed leaf alignment,
non-finite tensors, empty mask rows, invalid fold settings, and constant
validation targets with clear exceptions. They perform no persistence or
recovery from invalid input.

Source
------

.. literalinclude:: ../../../examples/single_tree_region_association.py
   :language: python
