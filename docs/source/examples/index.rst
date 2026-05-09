Examples
========

Run these examples from the repository root after installing the package in the
required environment. The examples use tiny local data, keep optional services
disabled by default, and only the TOML training workflow writes persistent
outputs under ``example_outputs/``.

.. toctree::
   :maxdepth: 1

   feature_engineering
   tree_to_graph
   tree_io
   single_task_training
   toml_training_config
   complete_pipeline

Recommended order
-----------------

1. :doc:`feature_engineering`
   Attaches node features to a small in-memory tree and prints a node-by-node
   feature summary.
2. :doc:`tree_to_graph`
   Converts a featured tree to graph tensors, then demonstrates virtual
   time-bin nodes.
3. :doc:`tree_io`
   Loads a repository sample tree when optional DendroPy dependencies are
   installed, or exits cleanly with installation guidance.
4. :doc:`single_task_training`
   Runs a compact single-task training loop with temporary output cleanup.
5. :doc:`toml_training_config`
   Trains a small configured model and writes a checkpoint and history under
   ``example_outputs/toml_training_config/``.
6. :doc:`complete_pipeline`
   Builds a tree, attaches features, converts it to graph data, loads a
   checkpoint or creates a temporary one, and prints a prediction.
