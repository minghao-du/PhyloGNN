User Guide
==========

Use these pages after the quickstart when you need workflow-level guidance for
real data preparation, training, and optional integrations.

Workflow pages
--------------

`Tree input <tree_input.html>`_
   Start from in-memory `ete3.Tree` objects or optional file readers.

`Feature engineering <feature_engineering.html>`_
   Attach deterministic numeric attributes to tree nodes.

`Graph conversion <graph_conversion.html>`_
   Convert feature-bearing trees into PyTorch Geometric `Data` objects.

`Datasets and splits <datasets_and_splits.html>`_
   Package graph samples, labels, and deterministic train/validation/test
   partitions.

`Training <training.html>`_
   Run the trainer lifecycle with PyG datasets, loaders, checkpoints, and
   predictions.

`Training configuration <training_config.html>`_
   Use local TOML files for repeatable model, trainer, loss, metrics, and
   tracking settings.

`Metrics and tracking <metrics_tracking.html>`_
   Use built-in metrics and optional Weights & Biases logging.

.. toctree::
   :maxdepth: 1

   tree_input
   feature_engineering
   graph_conversion
   datasets_and_splits
   training
   training_config
   metrics_tracking

How the pages fit
-----------------

Start with tree input, attach features, convert graphs, prepare datasets, and
train with local checkpoints. Optional pages explain tracking and file formats
that require extras. API details live in :doc:`../reference/index`.
