<!--
Sync Impact Report
Version change: template -> 1.0.0
Modified principles:
- Placeholder Principle 1 -> I. User-Oriented Simulation-Based Inference
- Placeholder Principle 2 -> II. Explicit Scientific and Data Contracts
- Placeholder Principle 3 -> III. Test-Backed Scientific Correctness
- Placeholder Principle 4 -> IV. Reproducible Workflow Integration
- Placeholder Principle 5 -> V. Pragmatic Simplicity Over Perfection
Added sections:
- Engineering Standards
- Delivery Workflow
Removed sections:
- None
Templates requiring updates:
- ✅ updated .specify/templates/plan-template.md
- ✅ updated .specify/templates/spec-template.md
- ✅ updated .specify/templates/tasks-template.md
- ⚠ pending .specify/templates/commands/*.md (directory not present in this repository)
- ⚠ pending runtime guidance docs such as README.md or docs/quickstart.md (not present in this repository)
Follow-up TODOs:
- None
-->
# PhyloGNN Constitution

## Core Principles

### I. User-Oriented Simulation-Based Inference
PhyloGNN MUST optimize for helping users perform simulation-based inference on
phylogenetic data with as little friction as practical. Features MUST serve a
real user workflow in the package, tests, or supported examples; speculative
abstractions, framework-heavy detours, and research-only side paths MUST NOT
displace delivery of usable SBI capabilities. Rationale: the package exists to
make SBI accessible, not to maximize architectural novelty.

### II. Explicit Scientific and Data Contracts
Public APIs, graph transformations, simulation interfaces, and workflow inputs
MUST declare their contracts explicitly through types, docstrings, validation,
and descriptive names. Code MUST fail fast on invalid shapes, dtypes, tree
state, configuration, or workflow parameters; silent coercion and stringly
typed control flow are prohibited unless an existing compatibility constraint is
documented. Rationale: scientific software becomes untrustworthy when data
contracts are implicit.

### III. Test-Backed Scientific Correctness
Every non-trivial behavioral change MUST include focused validation evidence in
the form of unit, regression, integration, or workflow tests, unless the change
is documentation-only and the omission is stated explicitly. Tests MUST prefer
real fixtures and realistic package behavior over mocks, and reviews MUST treat
scientific correctness, maintainability, and clean code concerns as first-class
quality gates. Rationale: PhyloGNN must be credible both as research software
and as a maintainable Python package.

### IV. Reproducible Workflow Integration
When a feature introduces or modifies pipeline behavior, the implementation MUST
keep package code and workflow orchestration cleanly separated, with
configuration externalized, outputs organized deterministically, and
documentation sufficient for another user to reproduce the run. Snakemake or
workflow-related assets MUST follow community best practices for structure,
validation, and small-scale testability. Rationale: reproducibility is a core
requirement for computational phylogenetics and SBI.

### V. Pragmatic Simplicity Over Perfection
Designs MUST start from the simplest approach that satisfies current user and
scientific requirements. Perfection MUST NOT be the enemy of good: teams MAY
defer polish, generalization, or automation when the deferral is explicit,
bounded, and does not compromise correctness, reproducibility, or user-facing
clarity. Complexity beyond the minimal viable design MUST be justified in the
plan. Rationale: sustained delivery matters more than speculative completeness.

## Engineering Standards

- Package code under `src/phylognn/` and tests under `tests/` are the source of
  truth when examples diverge.
- Public functions, methods, and classes MUST use explicit type hints and
  substantial docstrings when behavior, tensor shapes, graph fields, or feature
  semantics are non-trivial.
- Imports MUST be grouped consistently, names MUST reveal intent, and helpers
  SHOULD be small and composable rather than multiplexing unrelated concerns.
- Runtime validation MUST raise explicit `ValueError` or `TypeError` messages
  consistent with the existing codebase conventions.
- Graph and tensor code MUST preserve deterministic ordering whenever ordering
  affects features, traversal, batching, or reproducibility.
- Workflow configuration MUST use explicit files such as YAML where applicable;
  required parameters MUST be validated directly instead of hidden behind
  permissive defaults.

## Delivery Workflow

- Every spec MUST identify the target user workflow, the scientific or data
  contracts being introduced or changed, and the validation evidence required to
  trust the result.
- Every plan MUST pass the Constitution Check before research and design move
  forward; any violation requires a written justification in Complexity
  Tracking.
- Every task list for non-trivial work MUST include validation tasks, docstring
  or documentation updates where contracts change, and workflow documentation
  tasks when reproducibility is affected.
- Reviews MUST check for clean-code issues such as mixed responsibilities,
  unclear naming, duplication, hidden imports without justification, and overuse
  of mocks.
- When workflow assets are added, reviews MUST also verify reproducible layout,
  clear config ownership, and lightweight execution paths suitable for testing.

## Governance

This constitution supersedes local habits for planning and implementation within
this repository. Amendments MUST be recorded in `.specify/memory/constitution.md`
with a Sync Impact Report that identifies affected templates and follow-up work.
Versioning follows semantic intent: MAJOR for incompatible governance changes or
principle removals, MINOR for new principles or materially expanded obligations,
and PATCH for clarifications that do not change required behavior. Compliance
review is mandatory during planning, implementation, and review; if a change
cannot satisfy a principle, the exception and rationale MUST be written in the
relevant plan before implementation proceeds.

**Version**: 1.0.0 | **Ratified**: 2026-04-16 | **Last Amended**: 2026-04-16
