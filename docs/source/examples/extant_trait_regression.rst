Extant Trait Regression
=======================

Script: ``examples/extant_trait_regression.py``.

Inputs
------

- ``examples_data/carni70/carni70_tree.nwk`` containing the phylogenetic tree.
- ``examples_data/carni70/carni70_data.csv`` containing ``species``, ``size``,
  and ``range`` columns.
- ``GATNodeRegressor`` node features ``branch_length``, ``is_tip``, and log-range.

Run command
-----------

Run the example from the repository root:

.. code-block:: bash

   python examples/extant_trait_regression.py

Training workflow
-----------------

The script builds one graph in deterministic preorder and keeps the full
``x``/``edge_index`` topology visible to the node model.  It uses finite values
in ``data.y`` as the ``prediction_mask`` and partitions those nodes into
one-dimensional, boolean ``train_mask``, ``val_mask``, and ``test_mask``
splits.  The splits are disjoint, cover every finite target exactly, and each
contains at least one node.

Training uses the public ``TrainingConfig`` and ``Trainer`` APIs with these
fixed defaults:

- ``epochs=200`` and ``batch_size=1``
- ``learning_rate=1e-3`` and ``optimizer="adam"``
- automatic device selection and ``save_dir="example_outputs"``

The example adapts each train and validation mask to a single-item loader while
the ``Trainer`` owns optimizer steps, validation, ``train_loss``/``val_loss``
history, and the ``best_model.pt``/``final_model.pt``/``history.json`` files.
After ``Trainer.fit()`` completes, the example loads ``best_model.pt`` before
test evaluation and writes the compatibility checkpoint.

Evaluation and outputs
----------------------

Only ``test_mask`` nodes are evaluated.  Predictions and finite targets are
transformed with ``torch.expm1`` before calculating test MSE and R2 and before
creating the scatter plot; training and validation losses remain on the
log-transformed scale.

Expected output
---------------

Stable stdout markers include:

.. code-block:: text

   Epoch  10/200
   Extant trait regression summary
   train/val/test nodes:
   test MSE:
   test R2:
   checkpoint:
   loss plot:
   scatter plot:

Files written
-------------

- ``example_outputs/extant_trait_regression_best.pt``
- ``example_outputs/extant_trait_regression_loss.png``
- ``example_outputs/extant_trait_regression_scatter.png``

Repeated runs overwrite these three stable output paths.  The trainer's
additional lifecycle files remain in ``example_outputs/``.

Optional dependencies
---------------------

Install the ``examples`` extra for plotting support:

.. code-block:: bash

   python -m pip install -e ".[examples]"

Failure modes
-------------

- Missing tree or CSV files fail during input loading.
- Leaves without matching trait rows emit a warning and receive a ``NaN`` label.
- A tree with zero valid prediction nodes raises ``ValueError`` before training.

Source
------

.. literalinclude:: ../../../examples/extant_trait_regression.py
   :language: python
