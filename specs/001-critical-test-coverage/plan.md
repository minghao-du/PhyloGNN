# Implementation Plan: Critical Test Coverage

**Branch**: `001-critical-test-coverage` | **Date**: 2026-04-16 | **Spec**: [spec.md](/Users/Minghao/Research/PhyloGNN/specs/001-critical-test-coverage/spec.md)
**Input**: Feature specification from `/specs/001-critical-test-coverage/spec.md`

## Summary

Expand the PhyloGNN automated test suite so every public module and externally visible API under `src/phylognn/` is covered by focused success-path and failure-path tests, with stricter coverage expectations for core data/model/training code than for facade modules. The implementation will add or extend pytest modules, maintain a coverage-mapping artifact, and explicitly document any small set of justified manual exceptions tied to optional dependencies or non-deterministic behavior.

## Technical Context

**Language/Version**: Python >=3.8  
**Primary Dependencies**: PyTorch, PyTorch Geometric, ETE3, NumPy; optional DendroPy for BEAST/tree I/O paths  
**Storage**: Source files, pytest fixtures, and Markdown planning artifacts in the repository; no database  
**Testing**: pytest, import-surface tests, unit/regression tests, targeted workflow checks  
**Target Platform**: Local Python package development on macOS/Linux-style environments with optional scientific Python dependencies  
**Project Type**: Python library  
**Performance Goals**: Contributors can run targeted module-level tests quickly during iteration, while the full validation suite remains stable enough for pre-merge verification  
**Constraints**: Prefer realistic fixtures over heavy mocking; treat `src/phylognn/` and `tests/` as source of truth; examples are out of scope for complete coverage; optional-dependency paths may require explicit exception handling  
**Scale/Scope**: Cover all public modules and externally visible APIs under `src/phylognn/`, plus the coverage-mapping artifact and any limited exception records required to justify remaining gaps

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- `User Value`: Pass. The feature directly improves user trust in the package by catching regressions in public scientific workflows before release.
- `Scientific Contracts`: Public package exports, curated subpackage exports, graph conversion outputs, feature metadata surfaces, training configuration behavior, and optional tree I/O boundaries are all in scope. Invalid trees, tensor shapes/dtypes, missing attributes, and unsupported optional dependencies must continue to fail fast and be asserted by tests.
- `Validation Evidence`: The change requires focused pytest coverage for public exports, data conversion, feature engineering, models, training utilities, and optional I/O boundaries, plus a maintained coverage-mapping artifact. Any unautomated path must be documented as a small manual exception.
- `Reproducibility`: No workflow engine changes are planned. Reproducibility is handled through deterministic fixtures, stable export expectations, and repeatable local pytest commands captured in quickstart guidance.
- `Pragmatism`: Full example-script coverage is explicitly deferred. The plan also permits a small number of documented manual exceptions for paths blocked by optional dependencies or unstable external setup.

## Project Structure

### Documentation (this feature)

```text
specs/001-critical-test-coverage/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── coverage-mapping-contract.md
│   └── public-api-test-contract.md
└── tasks.md
```

### Source Code (repository root)

```text
src/
└── phylognn/
    ├── __init__.py
    ├── io.py
    ├── data/
    │   ├── __init__.py
    │   ├── converter.py
    │   ├── feature_engineer.py
    │   └── tree_io.py
    ├── models/
    │   ├── __init__.py
    │   ├── base.py
    │   ├── gat_lstm.py
    │   ├── layers.py
    │   └── multitask.py
    ├── training/
    │   ├── __init__.py
    │   ├── dataset.py
    │   ├── metrics.py
    │   └── trainer.py
    └── utils/
        ├── __init__.py
        └── tree_utils.py

tests/
├── conftest.py
├── test_data_conversion.py
├── test_feature_engineer.py
├── test_public_api.py
└── test_training_api.py
```

**Structure Decision**: Keep the existing single-package Python library structure. Add new or expanded pytest modules under `tests/` aligned to `phylognn.data`, `phylognn.models`, `phylognn.training`, `phylognn.io`, and top-level facade exports, and store design-time contracts in `specs/001-critical-test-coverage/contracts/`.

## Phase 0: Research Summary

Research decisions are recorded in [research.md](/Users/Minghao/Research/PhyloGNN/specs/001-critical-test-coverage/research.md). All technical unknowns for this feature are resolved without requiring additional clarification.

## Phase 1: Design Summary

- Define coverage entities, tiers, mapping records, and exception records in [data-model.md](/Users/Minghao/Research/PhyloGNN/specs/001-critical-test-coverage/data-model.md).
- Document the public API validation contract and the coverage-mapping artifact contract in `contracts/`.
- Provide contributor verification steps in [quickstart.md](/Users/Minghao/Research/PhyloGNN/specs/001-critical-test-coverage/quickstart.md).
- Update the Codex agent context after design artifacts are written.

## Post-Design Constitution Check

- `User Value`: Still passes. The design stays focused on release confidence and contributor safety, not on speculative tooling.
- `Scientific Contracts`: Still passes. The design explicitly names public exports, scientific graph/data contracts, and failure-mode assertions as required coverage targets.
- `Validation Evidence`: Still passes. The design requires automated tests plus a coverage mapping artifact and bounded exception documentation.
- `Reproducibility`: Still passes. The design relies on deterministic fixtures and repeatable local commands, with no hidden runtime state.
- `Pragmatism`: Still passes. The design narrows the scope to public modules and APIs, excludes examples from complete coverage, and bounds any unavoidable manual exceptions.

## Complexity Tracking

No constitution violations or extra complexity justifications are required for this plan.
