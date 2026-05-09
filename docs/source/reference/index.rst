Reference
=========

The reference documents curated public package surfaces, not every internal
module. Source code and tests are authoritative when examples disagree.

Fast import paths
-----------------

Use the root package for the main workflow objects:

.. code-block:: python

   from phylognn import (
       GATBiLSTMNet,
       TemporalBiLSTMEncoder,
       Trainer,
       TrainingConfig,
       TreeFeatureEngineer,
       TreeToGraphConverter,
       __version__,
   )

Reference areas
---------------

`Data <data.html>`_
   Feature engineering and tree-to-graph conversion APIs.

`Models <models.html>`_
   Public model classes and shared model base behavior.

`Training <training.html>`_
   Trainer, configuration, datasets, metrics, TOML helpers, and tracking.

`Tree I/O <io.html>`_
   Optional DendroPy-backed file reading and conversion helpers.

`Utilities <utils.html>`_
   Small public utility helpers.

.. toctree::
   :maxdepth: 1

   data
   models
   training
   io
   utils

Public coverage
---------------

The reference covers root exports, `phylognn.data`, `phylognn.models`,
`phylognn.training`, optional `phylognn.io`, and `phylognn.utils`. Low-level
model layers and private helpers are omitted unless they are intentionally
exposed by a curated public facade.
