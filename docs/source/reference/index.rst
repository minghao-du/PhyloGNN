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

:doc:`Data <data>`
   Feature engineering and tree-to-graph conversion APIs.

:doc:`Models <models>`
   Public model classes and shared model base behavior.

:doc:`Association <association>`
   In-memory single-tree region association evaluation and masked attention.

:doc:`Training <training>`
   Trainer, configuration, datasets, metrics, TOML helpers, and tracking.

:doc:`Tree I/O <io>`
   Optional DendroPy-backed file reading and conversion helpers.

:doc:`Utilities <utils>`
   Small public utility helpers.

.. toctree::
   :maxdepth: 1

   data
   models
   association
   training
   io
   utils

Public coverage
---------------

The reference covers root exports, `phylognn.data`, `phylognn.models`,
`phylognn.training`, optional `phylognn.io`, and `phylognn.utils`. Low-level
model layers and private helpers are omitted unless they are intentionally
exposed by a curated public facade.
