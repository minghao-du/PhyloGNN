<!--
Sync Impact Report
Version change: 2.0.0 -> 2.0.1
Modified principles:
- V. Environment-Constrained Minimal Delivery: required Conda environment corrected from `pytorch` to `phylognn`
Added sections:
- None
Removed sections:
- None
Templates requiring updates:
- ✅ updated .specify/templates/plan-template.md
- ✅ checked .specify/templates/spec-template.md
- ✅ checked .specify/templates/tasks-template.md
- ⚠ pending .specify/templates/commands/*.md (directory not present in this repository)
- ✅ checked AGENTS.md
Follow-up TODOs:
- None
-->
# PhyloGNN Constitution

## Core Principles

### I. Package Truth Over Examples
`src/phylognn/` and `tests/` MUST define correct behavior when examples or
ad hoc scripts diverge. New features, fixes, and refactors MUST strengthen the
Python package for converting phylogenetic trees into PyTorch Geometric data
and training GNN models on those graphs; example scripts MAY illustrate usage
but MUST NOT redefine package contracts. Rationale: repository drift is most
likely when examples become the de facto specification.

### II. Explicit Graph and Tensor Contracts
Code that constructs, mutates, loads, or consumes graph data MUST declare and
validate the required contract explicitly. Required fields such as `x`,
`edge_index`, `batch`, and task-specific attributes MUST be checked for shape,
dtype, and semantic readiness before use. Graph-level metadata MUST use
descriptive names that stay consistent with existing fields such as
`node_names`, `edge_type`, and `original_num_nodes`, and public APIs MUST
document these contracts through type hints and docstrings. Rationale:
scientific graph code is only trustworthy when data assumptions are visible and
enforced.

### III. Deterministic Phylogenetic Semantics
Whenever traversal order, feature order, batching, or serialization affects
outputs, the implementation MUST preserve deterministic behavior. Tree-to-graph
conversion, feature engineering, dataset indexing, and related helpers MUST
avoid hidden reordering, ambiguous naming, or nondeterministic iteration unless
the behavior is explicitly documented and tested. Rationale: reproducibility in
phylogenetic workflows depends on stable semantics, not just stable files.

### IV. Test-Backed Public Surface
Every non-trivial behavioral change MUST ship with focused tests that exercise
the affected package surface, using real fixtures and realistic `Data` objects
where practical. Public API changes MUST update docstrings, exports such as
`__all__`, and nearby tests; optional capability boundaries such as
`phylognn.io` MUST remain explicit and MUST NOT leak into default package
surfaces accidentally. Rationale: release safety depends on validating both
behavior and what the package chooses to expose.

### V. Environment-Constrained Minimal Delivery
Changes MUST fit the repository's current abstractions and execution
environment. Work MUST use the existing `phylognn` Conda environment on this
machine, MUST avoid unrelated refactors, and MUST prefer minimal local changes
over speculative architecture. New helpers, validators, and type aliases MUST
only be introduced after checking whether an equivalent already exists nearby.
Rationale: disciplined scope control keeps the package maintainable and reduces
scientific regression risk.

## Engineering Standards

- Package implementation lives under `src/phylognn/`, with `data/`, `models/`,
  `training/`, and `utils/` treated as the canonical architectural units.
- Tree, graph, and tensor code MUST raise explicit `ValueError` or `TypeError`
  messages when contracts are violated rather than relying on implicit failures.
- Public functions, methods, and classes MUST use explicit type hints, and
  non-trivial behavior MUST be documented with docstrings that describe graph
  fields, tensor shapes, and expected semantics.
- When adding graph-level metadata or labels, naming MUST remain descriptive
  and consistent with current package conventions.
- Optional dependencies and optional APIs MUST remain clearly separated from the
  default import surface so core usage does not depend on extra packages.

## Delivery Workflow

- Before editing, contributors MUST read the target module and nearby tests to
  understand the existing contract and avoid accidental surface changes.
- Plans and specs MUST identify which package modules, graph fields, dataset
  paths, or model interfaces are changing, and MUST state how determinism and
  validation will be preserved.
- Task lists for non-trivial work MUST include validation tasks and MUST include
  docstring or export updates whenever public behavior changes.
- Relevant tests MUST be run after code changes; narrow changes SHOULD start
  with the smallest affected test file, class, or function before broader
  verification.
- `ruff check` and `black --check` on touched areas SHOULD be run when
  practical, and any justified omission MUST be stated in the handoff.

## Governance

This constitution supersedes informal practice for planning, implementation,
and review in this repository. Amendments MUST be recorded in
`.specify/memory/constitution.md` together with a Sync Impact Report that lists
affected templates and any deferred follow-up. Versioning follows semantic
intent: MAJOR for incompatible principle redefinitions or removals, MINOR for
new principles or materially expanded obligations, and PATCH for clarifications
that preserve the same required behavior. Compliance review is mandatory during
planning, implementation, and review; if a change cannot satisfy a principle,
the exception and rationale MUST be documented before implementation proceeds.

**Version**: 2.0.1 | **Ratified**: 2026-04-16 | **Last Amended**: 2026-04-26
