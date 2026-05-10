Quickstart
==========

This first tutorial creates an `ete3.Tree`, attaches node features, converts it
to a PyTorch Geometric `Data` object, adds a target label, runs a tiny training
smoke test, and prints a prediction.

Create a small tree
-------------------

.. literalinclude:: ../../examples/quickstart_training.py
   :language: python
   :start-after: [START build_tree]
   :end-before: [END build_tree]

Attach node features
--------------------

`TreeFeatureEngineer` writes numeric attributes to each tree node. Use
`feature_names` as the stable column order for graph conversion.

.. literalinclude:: ../../examples/quickstart_training.py
   :language: python
   :start-after: [START make_graph]
   :end-before: [END make_graph]

Convert the tree to graph data
------------------------------

`TreeToGraphConverter` reads node attributes into graph tensors. The same
snippet above also adds a dummy graph-level target label as `data.y`, which is
the field the trainer expects during supervised training.

Add a target label
------------------

The smoke test uses a single regression target:

.. code-block:: python

   data.y = torch.tensor([1.0], dtype=torch.float32)

For real datasets, attach one target per graph and keep target shape compatible
with the selected model head and loss.

Validate the graph fields
-------------------------

Before training, check the required tensor shapes and dtypes.

.. literalinclude:: ../../examples/quickstart_training.py
   :language: python
   :start-after: [START validate_graph]
   :end-before: [END validate_graph]

For complete field semantics, including `data.x`, `data.edge_index`,
`data.edge_type`, `data.time_bin`, and deterministic node ordering, see
:doc:`user_guide/graph_conversion`.

Run a tiny training smoke test
------------------------------

Run the maintained script from the repository root:

.. code-block:: console

   python examples/quickstart_training.py

The training function creates a temporary output directory, trains for two
epochs on the one-graph dataset, and returns one prediction.

.. literalinclude:: ../../examples/quickstart_training.py
   :language: python
   :start-after: [START train_and_predict]
   :end-before: [END train_and_predict]

Expected output includes stable markers like these:

.. code-block:: text

   Quickstart training summary
   x shape:
   edge_index shape:
   target shape:
   batch ready: true
   prediction:

Completion summary
------------------

At this point you have created a tree, attached deterministic features,
converted it to graph data, validated the required fields, trained a tiny
model, and printed a prediction.

Next steps
----------

.. list-table::
   :header-rows: 1

   * - Need
     - Go to
   * - Prepare real trees and features
     - :doc:`user_guide/index`
   * - Understand graph fields
     - :doc:`user_guide/graph_conversion`
   * - Configure datasets, splits, and TOML training
     - :doc:`user_guide/training_config`
   * - Run complete scripts
     - :doc:`examples/index`
