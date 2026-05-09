Examples
========

Run these examples from the repository root after installing the package in the
required environment. The examples use tiny local data, write outputs under
``example_outputs/``, and keep optional services disabled by default.

.. toctree::
   :maxdepth: 1

   toml_training_config
   complete_pipeline

Recommended order
-----------------

1. :doc:`toml_training_config`
   Trains a small configured model and writes the checkpoint used by the
   pipeline example.
2. :doc:`complete_pipeline`
   Builds a tree, attaches features, converts it to graph data, loads the
   checkpoint, and prints a prediction.

