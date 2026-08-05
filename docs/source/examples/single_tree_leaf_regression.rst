Single-Tree Leaf Regression
===========================

Script: ``examples/single_tree_leaf_regression.py``.

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

   python examples/single_tree_leaf_regression.py

Expected output
---------------

The deterministic transductive workflow prints a compact summary:

.. code-block:: text

   Single-tree leaf regression summary
   leaf count:
   fold scores:
   overall score:
   OOF predictions shape:
   final predictions:
   attention summary:

The displayed scores are a deterministic demonstration, not a biological
claim. The attention summary identifies the position with the largest mean
valid-position attention weight when the model supplies attention.

Files written
-------------

None. The tree, tensors, models, folds, scores, predictions, and attention
remain in memory.

Optional dependencies
---------------------

None. The example needs only the core package dependencies, including PyTorch
and ETE3.

Failure modes
-------------

The workflow rejects malformed leaf alignment, non-finite tensors, empty mask
rows, invalid fold settings, and undefined default R-squared folds with clear
exceptions. It performs no persistence or recovery from invalid input.

Source
------

.. literalinclude:: ../../../examples/single_tree_leaf_regression.py
   :language: python
