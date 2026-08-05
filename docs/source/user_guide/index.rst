User Guide
==========

Use these pages after the quickstart when you need workflow-level guidance for
real data preparation, training, and optional integrations.

Workflow pages
--------------

:doc:`Tree input <tree_input>`
   Start from in-memory `ete3.Tree` objects or optional file readers.

:doc:`Leaf regression <leaf_regression>`
   Predict one continuous target per masked leaf with deterministic validation.

:doc:`Feature engineering <feature_engineering>`
   Attach deterministic numeric attributes to tree nodes.

:doc:`Graph conversion <graph_conversion>`
   Convert feature-bearing trees into PyTorch Geometric `Data` objects.

:doc:`Datasets and splits <datasets_and_splits>`
   Package graph samples, labels, and deterministic train/validation/test
   partitions.

:doc:`Training <training>`
   Run the trainer lifecycle with PyG datasets, loaders, checkpoints, and
   predictions.

:doc:`Training configuration <training_config>`
   Use local TOML files for repeatable model, trainer, loss, metrics, and
   tracking settings.

:doc:`Metrics and tracking <metrics_tracking>`
   Use built-in metrics and optional Weights & Biases logging.

.. toctree::
   :maxdepth: 1

   tree_input
   leaf_regression
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
