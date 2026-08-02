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

The script builds one graph, masks valid extant labels into train/validation/test
sets, trains for 200 epochs, and evaluates size after reversing the log transform.

Expected output
---------------

Stable stdout markers include:

.. code-block:: text

   Epoch  10/200
   Extant trait regression summary
   test MSE:
   test R2:

Files written
-------------

- ``example_outputs/extant_trait_regression_best.pt``
- ``example_outputs/extant_trait_regression_loss.png``
- ``example_outputs/extant_trait_regression_scatter.png``

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
