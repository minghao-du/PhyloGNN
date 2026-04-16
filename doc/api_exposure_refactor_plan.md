# PhyloGNN API Exposure Refactor Plan

## Goal

This document proposes a concrete cleanup plan for the public API currently
exposed from `src/phylognn/`.

The main objective is to make the user-facing API:

- easier to understand
- cheaper to maintain
- less fragile under internal refactors
- more explicit about what is stable vs. internal


## Current Assessment

From the current `src/phylognn/` package layout, the API is partially coherent,
but the public boundary is not fully controlled.

What already works well:

- The core data pipeline is clear:
  `TreeFeatureEngineer -> TreeToGraphConverter`
- The package already has meaningful subpackages:
  `data`, `models`, `training`, `utils`
- Several modules contain substantial docstrings and explicit validation logic

Main issues:

1. Root import is too tightly coupled to optional tree I/O code.
2. `training/__init__.py` does not define a correct and trustworthy package API.
3. `models/__init__.py` exports both end-user models and low-level building
   blocks at the same level.
4. Some public contracts are inconsistent or overly mutable.
5. The package does not clearly distinguish:
   stable high-level API, advanced subpackage API, and internal implementation API.


## Design Principle

Adopt a three-layer API model.

### Layer 1: Stable Top-Level API

Expose only the objects that you are willing to support as the primary user
entry points:

- `TreeFeatureEngineer`
- `TreeToGraphConverter`
- `read_tree_as_ete3` or a renamed tree-loading function if you decide tree I/O
  is part of the official product surface
- `TrainingConfig`
- `Trainer`
- `GATBiLSTMNet`
- `MultiTaskGATNet`

This layer should optimize for discoverability and long-term stability.

### Layer 2: Advanced Subpackage API

Keep richer but still intentional APIs under:

- `phylognn.data`
- `phylognn.models`
- `phylognn.training`

These are for users who need more control, but the exported names should still
be curated.

### Layer 3: Internal / Low-Level API

Leave implementation details importable only from explicit module paths, not
from package-level `__init__.py` files.

Examples:

- `phylognn.models.layers.GATBlock`
- `phylognn.models.layers.MLPHead`
- `phylognn.models.multitask.TaskHead`
- low-level helpers in `training.dataset`, `training.trainer`, `data.tree_io`

This preserves flexibility for future refactors.


## Recommended Target API

### `phylognn`

Recommended exports:

```python
from .data import TreeFeatureEngineer, TreeToGraphConverter
from .training import Trainer, TrainingConfig
from .models import GATBiLSTMNet, MultiTaskGATNet
```

Optional:

```python
from .io import read_tree_as_ete3
```

Recommended `__all__`:

```python
__all__ = [
    "TreeFeatureEngineer",
    "TreeToGraphConverter",
    "Trainer",
    "TrainingConfig",
    "GATBiLSTMNet",
    "MultiTaskGATNet",
]
```

If you want tree I/O as a first-class API, then add `read_tree_as_ete3`.

Also rename:

```python
version = "0.1.0"
```

to:

```python
__version__ = "0.1.0"
```

This is the more standard package contract.


## Subpackage Recommendations

### 1. `phylognn.data`

Recommended exports:

- `TreeFeatureEngineer`
- `TreeToGraphConverter`

Optional export:

- `read_tree_as_ete3`

Recommended rule:

- If `dendropy` is meant to be optional, do not import tree I/O at package
  import time.
- Move tree-loading helpers behind a dedicated module boundary such as
  `phylognn.io` or `phylognn.data.io`.

Preferred structure:

```python
phylognn.data
    TreeFeatureEngineer
    TreeToGraphConverter

phylognn.io
    read_tree_as_ete3
    TreeReadConfig
```

Why:

- users who only do feature engineering / graph conversion should not be forced
  through an optional file-parsing dependency chain
- tree I/O is conceptually different from feature engineering and conversion


### 2. `phylognn.training`

This package should expose a clean and complete training API.

Recommended exports:

- `DatasetSplit`
- `SplitDatasetView`
- `SplitPhyloDataset`
- `SplitPhyloDiskDataset`
- `Trainer`
- `TrainingConfig`
- `create_default_trainer`
- `mse_metric`
- `mae_metric`
- `r2_metric`
- `rmse_metric`
- `relative_error_metric`

Required fixes:

- replace `all = [...]` with `__all__ = [...]`
- remove nonexistent `PhyloDataset`
- export `create_default_trainer`
- export all metric helpers that are intended to be public

Suggested `__all__`:

```python
__all__ = [
    "DatasetSplit",
    "SplitDatasetView",
    "SplitPhyloDataset",
    "SplitPhyloDiskDataset",
    "Trainer",
    "TrainingConfig",
    "create_default_trainer",
    "mse_metric",
    "mae_metric",
    "r2_metric",
    "rmse_metric",
    "relative_error_metric",
]
```


### 3. `phylognn.models`

Recommended package-level exports:

- `BasePhyloGNN`
- `BaseGATNet`
- `GATBiLSTMNet`
- `MultiTaskGATNet`

Do not package-export by default:

- `GATBlock`
- `ResidualGATStack`
- `PositionalEncoding`
- `MLPHead`
- `TaskHead`

Reason:

- these are building blocks, not the primary user-facing modeling API
- once exported at package level, they become much harder to redesign safely

If advanced users need them, they can still import from concrete module paths.


### 4. `phylognn.utils`

Current utility surface is too thin and too domain-specific to justify strong
package-level exposure.

Recommendation:

- either keep `utils` minimal and explicitly internal
- or rename/promote utilities only when they are broadly reusable and stable

For now, avoid pushing `utils` objects into the root package API.


## Specific Contract Fixes

### 1. Make public metadata less mutable

Current issue:

- `TreeFeatureEngineer.feature_names` is a mutable list
- `TreeFeatureEngineer.available_features` is a mutable set

This weakens the boundary of the public API.

Recommended change:

- store internal mutable state privately
- expose read-only views through properties

Preferred pattern:

```python
@property
def feature_names(self) -> Tuple[str, ...]:
    return tuple(self._feature_registry.keys())

@property
def available_features(self) -> FrozenSet[str]:
    return frozenset(self._feature_registry.keys())
```

This keeps downstream usage stable while preventing accidental mutation.


### 2. Unify virtual-node feature naming

Current issue:

- converter docs describe `"is_virtual_node"`
- `output_feature_names` returns `"is_virtual"`

Recommendation:

- pick one name and use it everywhere
- prefer the more explicit `"is_virtual_node"`

Then align:

- `output_feature_names`
- conversion logic
- docstrings
- any downstream assumptions


### 3. Standardize public naming conventions

Recommended conventions:

- public package version: `__version__`
- public constants only when users need them
- helper factories included in exports only if intentionally supported

Avoid exposing names implicitly just because they exist in module scope.


## Refactor Strategy

### Phase 1: Safe API Cleanup

Low-risk changes with immediate benefit:

1. Fix all `__all__` definitions.
2. Remove nonexistent exports.
3. Add missing public exports that are already intended for use.
4. Rename `version` to `__version__`.
5. Fix inconsistent contract names such as `is_virtual` vs.
   `is_virtual_node`.

This phase should not require architectural changes.


### Phase 2: Reduce Overexposure

1. Remove low-level layer classes from `phylognn.models.__all__`.
2. Keep those classes importable only from explicit module paths.
3. Decide whether `TaskHead` is internal-only.

This phase tightens the public boundary without changing core functionality.


### Phase 3: Decouple I/O From Core Data API

1. Move tree loading into a separate package boundary:
   `phylognn.io` or `phylognn.data.io`.
2. Prevent root package import from requiring tree I/O dependencies.
3. Optionally make `dendropy` an optional extra if that matches your packaging
   goals.

This is the most meaningful design improvement for maintainability.


## Proposed File-Level Changes

### `src/phylognn/__init__.py`

Change from a minimal preprocessing-only root API to an intentional top-level
workflow API.

Recommended responsibilities:

- expose only stable high-level classes/functions
- define `__version__`
- avoid importing optional or heavy dependencies unless they are truly part of
  the default contract


### `src/phylognn/data/__init__.py`

Recommended responsibilities:

- expose only the primary data pipeline objects
- avoid automatic import of optional tree I/O unless you explicitly want it as
  part of the default package behavior


### `src/phylognn/models/__init__.py`

Recommended responsibilities:

- expose final user-facing models
- optionally expose abstract bases
- avoid exporting low-level architectural parts


### `src/phylognn/training/__init__.py`

Recommended responsibilities:

- provide a complete and correct package API
- ensure exports match actual symbols
- include the trainer factory if it is meant for users


## Suggested Final Import Experience

### Common users

```python
from phylognn import (
    TreeFeatureEngineer,
    TreeToGraphConverter,
    TrainingConfig,
    Trainer,
    GATBiLSTMNet,
    MultiTaskGATNet,
)
```

### Advanced data users

```python
from phylognn.data import TreeFeatureEngineer, TreeToGraphConverter
from phylognn.io import read_tree_as_ete3
```

### Advanced model users

```python
from phylognn.models import GATBiLSTMNet, MultiTaskGATNet
from phylognn.models.base import BaseGATNet
from phylognn.models.layers import GATBlock
```

### Training users

```python
from phylognn.training import (
    DatasetSplit,
    SplitPhyloDataset,
    SplitPhyloDiskDataset,
    TrainingConfig,
    Trainer,
    create_default_trainer,
)
```


## What Not To Do

- Do not expose every class just because it is potentially reusable.
- Do not mix internal building blocks and stable end-user API at the same
  package level.
- Do not let package-level imports require optional subsystems unless that is a
  deliberate product decision.
- Do not keep mutable public metadata objects when read-only contracts are
  sufficient.


## Recommended Priority Order

1. Fix `training/__init__.py`.
2. Tighten `models/__init__.py`.
3. Standardize root package exports and `__version__`.
4. Fix public contract inconsistencies in `TreeToGraphConverter` and
   `TreeFeatureEngineer`.
5. Split tree I/O away from the default import path.


## Final Recommendation

Your current API is usable, but it still behaves like a codebase that has grown
module-by-module rather than a package with a deliberately curated public
surface.

The strongest part of the current design is the core preprocessing pipeline.
That should remain the center of the package.

The most valuable next step is not adding more API, but reducing ambiguity:

- clearly define the stable top-level API
- make subpackage exports intentional
- stop exporting low-level implementation details by default
- decouple optional tree I/O from the core import path

If you later want, a follow-up document can specify the exact patch plan for
each `__init__.py` and the minimal backward-compatible migration path.
