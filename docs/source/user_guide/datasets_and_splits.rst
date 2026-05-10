Datasets And Splits
===================

PhyloGNN training utilities work with PyTorch Geometric `Data` samples. Build
graphs first, attach each graph target as `data.y`, then choose an in-memory or
disk-backed dataset depending on how many samples you need to manage.

Dataset choices
---------------

.. list-table::
   :header-rows: 1

   * - Situation
     - Use
     - Notes
   * - Small or generated datasets
     - `SplitPhyloDataset`
     - Keeps `Data` objects and labels in memory.
   * - Preprocessed graph files
     - `SplitPhyloDiskDataset`
     - Reads `.pt` graphs, with optional mirrored label files.
   * - Existing train/validation/test IDs
     - `DatasetSplit.from_manifest_dir()`
     - Loads `train.txt`, `val.txt`, and `test.txt`.

Split definitions
-----------------

`DatasetSplit.from_ratios()` creates deterministic train, validation, and test
IDs from sample IDs, ratios, and a seed. Use this when you need a repeatable
local split without maintaining manifest files.

`DatasetSplit.from_dict()` accepts explicit sample IDs for each split. Use this
when splits come from previous experiments or external metadata.

`SplitPhyloDataset.build_subsets(split)` returns split-specific views that can
be passed directly to `Trainer.fit()` and `Trainer.predict()`.

Target labels
-------------

Each supervised sample must expose `data.y`. Keep label dtype and shape aligned
with the selected loss and model output. For graph-level regression, a common
shape is one float target per graph.

Disk-backed `.pt` graph and label files are loaded as complete PyTorch
objects. Use this path only for trusted artifacts produced by your PhyloGNN
preprocessing workflow.

Determinism
-----------

Keep sample IDs stable across preprocessing runs. Use explicit feature order
during graph conversion so every split sees the same column layout. Use a fixed
split seed when ratio-based splits are acceptable.

Related pages
-------------

See :doc:`graph_conversion` for graph field contracts,
:doc:`training` for trainer usage, and :doc:`../reference/training` for API
lookup.
