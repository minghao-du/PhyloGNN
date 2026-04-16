# Contract: Public API Surface

## Purpose

Define the intended public import surfaces for the API exposure refactor. This
is a library-interface contract rather than an HTTP or CLI contract.

## Layer 1: Stable Top-Level API

The root package MUST expose only the primary workflow entry points that common
PhyloGNN users are expected to rely on.

### Required root-package exports

- `TreeFeatureEngineer`
- `TreeToGraphConverter`
- `TrainingConfig`
- `Trainer`
- `GATBiLSTMNet`
- `MultiTaskGATNet`
- `__version__`

### Root-package guarantees

- Imports for the names above MUST resolve successfully.
- The root package MUST NOT expose low-level model-layer helpers as stable API.
- The root package SHOULD avoid importing optional tree I/O by default.

## Layer 2: Advanced Subpackage APIs

### `phylognn.data`

Required curated exports:

- `TreeFeatureEngineer`
- `TreeToGraphConverter`

Optional intentional export:

- tree-loading functionality through the dedicated `phylognn.io` module

Rules:

- Data-pipeline usage MUST remain discoverable.
- Optional tree I/O MUST be intentionally separated from the default root
  import path.

### `phylognn.models`

Required curated exports:

- `BasePhyloGNN`
- `BaseGATNet`
- `GATBiLSTMNet`
- `MultiTaskGATNet`

Not package-exported by default:

- `GATBlock`
- `ResidualGATStack`
- `PositionalEncoding`
- `MLPHead`
- `TaskHead`

### `phylognn.training`

Required curated exports:

- `DatasetSplit`
- `SplitDatasetView`
- `SplitPhyloDataset`
- `SplitPhyloDiskDataset`
- `Trainer`
- `TrainingConfig`
- `create_default_trainer` if it is an intentional user-facing helper
- `mse_metric`
- `mae_metric`
- `r2_metric`
- `rmse_metric`
- `relative_error_metric`

Rules:

- Export declarations MUST use canonical Python package conventions.
- Nonexistent names MUST NOT appear in the public training API.

## Layer 3: Internal Module-Path APIs

The following categories remain importable only from explicit module paths and
MUST NOT be presented as stable package-level API unless deliberately promoted:

- low-level model layers
- task-specific heads
- internal helper utilities
- training internals not intended for end users
- low-level tree I/O implementation helpers

## Public Naming Contract

- Package version metadata MUST use `__version__`.
- The virtual-node feature flag MUST use one canonical public name throughout
  outputs, docs, and tests.
- Read-only public feature metadata MUST expose deterministic content.

## Validation Expectations

- Package-level import tests MUST cover the required root exports.
- Subpackage import tests MUST cover the curated advanced APIs.
- Regression tests MUST verify that accidental or nonexistent exports are not
  represented as part of the public contract.
