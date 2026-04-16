# Quickstart: API Exposure Refactor

## Goal

Validate that the refactored package exposes a clear stable API while keeping
the current PhyloGNN framework and PyTorch plus PyTorch Geometric stack.

## Prerequisites

1. Create and activate a virtual environment.
2. Install the package in editable mode with development dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Verify Stable Top-Level Imports

Run a Python shell and verify the stable workflow entry points:

```python
from phylognn import (
    TreeFeatureEngineer,
    TreeToGraphConverter,
    TrainingConfig,
    Trainer,
    GATBiLSTMNet,
    MultiTaskGATNet,
    __version__,
)
```

Expected outcome:

- all imports succeed
- `__version__` is defined
- no low-level model-layer classes are needed for the common workflow

## Verify Advanced Subpackage Imports

```python
from phylognn.data import TreeFeatureEngineer, TreeToGraphConverter
from phylognn.models import BasePhyloGNN, BaseGATNet, GATBiLSTMNet, MultiTaskGATNet
from phylognn.training import (
    DatasetSplit,
    SplitDatasetView,
    SplitPhyloDataset,
    SplitPhyloDiskDataset,
    Trainer,
    TrainingConfig,
    mse_metric,
    mae_metric,
    r2_metric,
    rmse_metric,
    relative_error_metric,
)
```

Expected outcome:

- advanced imports succeed from curated subpackages
- unsupported low-level symbols are not required at package level

## Verify Optional Tree I/O Boundary

```python
from phylognn.io import TreeReadConfig, read_tree_as_ete3
```

Expected outcome:

- optional tree-loading helpers are available from `phylognn.io`
- the default `phylognn` and `phylognn.data` import surfaces do not require
  tree I/O

## Verify Core Workflow Without Optional Tree I/O

Use the core preprocessing and modeling workflow without calling tree-loading
helpers:

```python
from ete3 import Tree
from phylognn import TreeFeatureEngineer, TreeToGraphConverter

tree = Tree("((A:1,B:2)C:3,D:4)E:0;", format=1)
engineer = TreeFeatureEngineer(num_time_bins=10)
tree = engineer.add_features(tree, origin_time=10.0)

converter = TreeToGraphConverter(
    feature_names=engineer.feature_names,
    num_time_bins=10,
)
data = converter.convert(tree)
```

Expected outcome:

- the core workflow succeeds using the stable package surface
- optional tree I/O is not required for this workflow

## Run Focused Validation Tests

```bash
pytest tests/test_feature_engineer.py tests/test_data_conversion.py
```

Then add or run focused API-surface tests for:

- root-package exports
- curated subpackage exports
- public metadata naming
- import behavior without optional tree I/O coupling
