# Examples Redesign Spec

**Date**: 2026-04-17
**Status**: Proposed
**Scope**: Rebuild `examples/` as a documentation-first, single-task-focused example suite for the current public PhyloGNN API.

## Context

The repository currently contains several example scripts under `examples/`, but they do not yet function as a coherent user-facing entry point:

- naming is inconsistent
- some scripts mix multiple tutorial goals
- some content appears partially unfinished
- script behavior does not consistently reflect the curated public API and current package contracts

The package and tests have recently moved toward clearer public facades under `phylognn`, `phylognn.data`, `phylognn.training`, and `phylognn.io`. The example suite should match that direction and help a user understand the recommended usage paths without forcing them to reverse-engineer internals.

This spec follows the repository constitution, especially:

- explicit scientific and data contracts
- user-oriented workflows
- pragmatic simplicity over perfection
- examples and docs deferring to `src/phylognn/` and `tests/` as the source of truth

## Goals

- Provide a clear, low-maintenance example suite that acts as the primary user entry point.
- Prefer documentation-first navigation with a small number of scripts over a large collection of partially overlapping demos.
- Cover the core user learning path:
  - feature engineering
  - tree-to-graph conversion
  - optional tree file I/O
  - single-task model training
- Keep examples aligned with supported public APIs and current behavior contracts.
- Favor examples that are easy to run locally and easy to understand.

## Non-Goals

- Exhaustively demonstrate every parameter or every supported API combination.
- Preserve old example filenames, structure, or script behavior for compatibility.
- Provide a multi-task training example in this redesign.
- Turn examples into a test suite replacement or a workflow orchestration framework.
- Optimize model quality, training duration, or scientific benchmarking in example scripts.

## User Experience Principles

- A new user should know which example to run first without guessing.
- Each script should answer one primary question.
- Scripts should be short enough to read in one pass.
- Longer explanations, prerequisites, and navigation should live in `examples/README.md`, not be duplicated across scripts.
- Examples should be useful even if they are not exhaustive.

## Proposed Example Structure

The redesigned `examples/` directory should contain the following files:

### `examples/README.md`

The main navigation document for all examples.

Responsibilities:

- explain the purpose of the example suite
- define the recommended reading and running order
- distinguish self-contained demos from examples that rely on repository sample data
- list prerequisites and optional dependencies
- provide one-line summaries and run commands for each script

### `examples/feature_engineering.py`

A self-contained introductory example for `TreeFeatureEngineer`.

Responsibilities:

- construct a small in-memory `ete3.Tree`
- initialize `TreeFeatureEngineer` using the recommended public API
- add a representative set of features
- print a compact summary of node-level feature values

Intentional exclusions:

- no exhaustive feature enumeration
- no complex custom feature extension path
- no file I/O
- no training

### `examples/tree_to_graph.py`

A self-contained example for `TreeToGraphConverter`.

Responsibilities:

- start from a small tree with engineered features
- initialize `TreeToGraphConverter` using the recommended public API
- convert the tree to a PyTorch Geometric `Data` object
- print the graph contract at a high-signal level, such as:
  - `x` shape
  - `edge_index` shape
  - number of nodes and edges
  - selected metadata fields when present

Intentional exclusions:

- no broad option matrix
- no multi-branch converter tutorial
- no training

### `examples/tree_io.py`

An example for the optional tree I/O boundary.

Responsibilities:

- demonstrate the supported `phylognn.io` entry point
- read a tree from repository sample data under `examples_data/`
- show how the loaded tree can feed into the rest of the package workflow
- clearly state that this is an optional path and may require optional dependencies

Intentional exclusions:

- no deep parser internals
- no mixed training or disk workflow logic
- no attempt to replace API reference documentation

### `examples/single_task_training.py`

The only training example in scope for this redesign.

Responsibilities:

- serve as the main end-to-end example
- demonstrate a complete single-task path:
  - tree creation or loading
  - feature engineering
  - graph conversion
  - synthetic or lightweight example label construction
  - train/validation/test split
  - model initialization
  - trainer configuration
  - short training run
  - basic evaluation or prediction summary

Constraints:

- keep dataset size and epochs small enough for example use
- favor clarity over scientific realism
- use a single-task workflow only

Intentional exclusions:

- no multi-task path
- no large hyperparameter surface
- no attempt to demonstrate production-scale experiment management

### `examples/full_pipeline.py`

This file is optional and should only remain if it has a unique responsibility not already covered by `single_task_training.py`.

Allowed reason to keep it:

- it demonstrates a repository-data-based disk workflow that is meaningfully different from the single-task training example

Reason to remove it:

- it duplicates the role of the main training example
- it creates multiple competing "main entry points" for users

Decision rule:

- keep only one primary end-to-end entry point for training
- remove `full_pipeline.py` if its behavior substantially overlaps with `single_task_training.py`

## Naming And Style Standards

- Use clear English filenames based on responsibility.
- Do not preserve historical names such as `examples_*.py` unless they still precisely describe the file.
- Each script should provide a `main()` entry point.
- Each script should be runnable directly with `python examples/<name>.py`.
- Script output should be concise and high-signal.
- Code comments should remain sparse; explanation belongs mostly in `examples/README.md`.
- Example code should use public APIs rather than internal implementation details wherever possible.

## Documentation Strategy

`examples/README.md` should become the canonical navigation layer for examples.

It should include:

- a short explanation of what PhyloGNN examples are for
- installation notes for core and optional dependencies
- a recommended order such as:
  1. `feature_engineering.py`
  2. `tree_to_graph.py`
  3. `tree_io.py`
  4. `single_task_training.py`
- exact run commands
- a short explanation of expected outputs
- notes on which examples are self-contained and which rely on `examples_data/`

The README should not become a full tutorial book. It should provide navigation and the minimum framing needed to make the scripts useful.

## Data And Dependency Policy

- Self-contained teaching examples should not require external user-supplied data files.
- The I/O example and the end-to-end training example may depend on repository sample data under `examples_data/`.
- Optional dependency requirements must be stated clearly in the examples README and in the relevant script docstring when needed.
- Output directories, if used by an example, must be explicit and easy to clean up.

## API Alignment Requirements

The example suite must align with the current supported package contracts.

In practice:

- examples should prefer imports from curated public surfaces
- optional tree-loading helpers should come from `phylognn.io`
- if an older example conflicts with current package behavior, the example must be updated or removed
- package source and current tests take precedence over old example behavior

## Migration Strategy

The redesign may fully replace the existing `examples/` contents.

Migration approach:

1. Define the new target file set first.
2. Review each existing example only for reusable logic or useful explanatory content.
3. Delete or replace scripts that do not fit the new structure.
4. Avoid carrying forward complexity that exists only because of historical example drift.

The redesign should treat old examples as reference material, not as compatibility constraints.

## Verification Expectations

The implementation should verify the example suite with pragmatic checks:

- each example script should be run locally at least once
- commands listed in `examples/README.md` must match the actual filenames
- examples should complete successfully within reasonable example-scale runtime expectations
- examples should not claim unsupported behavior

This redesign does not require perfection before delivery. It requires that examples become clear, runnable, and maintainable for the current public API.

## Implementation Notes For Planning

The eventual implementation plan should account for:

- deciding whether `full_pipeline.py` remains or is deleted
- replacing inconsistent historical names with clear English names
- writing `examples/README.md` as the single navigation entry point
- simplifying example outputs
- verifying that training example settings remain lightweight
- ensuring example imports stay on public package boundaries

## Open Decisions Already Resolved

- The suite should be documentation-first rather than script-count-first.
- The suite should support both modular demos and a single end-to-end training path.
- Training examples should cover single-task workflows only.
- Old examples may be fully deleted if they are not suitable.
- The package should avoid letting perfection block a good, usable example suite.

## Acceptance Summary

This design is successful when:

- a user can open `examples/README.md` and immediately understand where to start
- each retained script has one clear responsibility
- there is exactly one primary single-task training example
- optional I/O remains explicitly separated from the core self-contained demos
- examples reflect the current supported public API rather than stale historical usage
