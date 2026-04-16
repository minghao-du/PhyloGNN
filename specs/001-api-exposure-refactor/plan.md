# Implementation Plan: API Exposure Refactor

**Branch**: `001-api-exposure-refactor` | **Date**: 2026-04-16 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-api-exposure-refactor/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/plan-template.md` for the execution workflow.

## Summary

Curate PhyloGNN's public API into three intentional layers: a stable top-level
workflow API, curated advanced subpackage APIs, and explicit module-path-only
internal APIs. The implementation keeps the existing Python package layout and
current PyTorch plus PyTorch Geometric stack, while tightening `__init__.py`
exports, decoupling optional tree I/O from the default import path, and making
user-visible metadata and naming contracts consistent.

## Technical Context

<!--
  ACTION REQUIRED: Replace the content in this section with the technical details
  for the project. The structure here is presented in advisory capacity to guide
  the iteration process.
-->

**Language/Version**: Python >=3.8  
**Primary Dependencies**: PyTorch, PyTorch Geometric, ETE3, NumPy; optional tree I/O path may use DendroPy via extras  
**Storage**: Source files, package metadata, and optional serialized `.pt` graph artifacts; no database  
**Testing**: pytest for unit and regression tests; import-surface and API-contract validation tests will be added  
**Target Platform**: Python package on macOS/Linux development environments and general research compute environments  
**Project Type**: Python library  
**Performance Goals**: Preserve current import and runtime behavior for core workflows; avoid adding overhead or new required dependencies to default imports  
**Constraints**: Keep the existing package framework; use the current PyTorch and PyTorch Geometric stack; preserve documented primary workflows; fail fast on invalid public contracts; avoid forcing optional tree I/O dependencies into the root import path  
**Scale/Scope**: Refactor package export surfaces across `src/phylognn/__init__.py`, `data`, `models`, and `training`, plus aligned tests and documentation/examples

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- `User Value`: Pass. The feature improves import discoverability and
  maintainability for real package users performing feature engineering, graph
  conversion, model use, and training.
- `Scientific Contracts`: Pass. The affected contracts are package import
  surfaces, export lists, public metadata (`__version__`), feature-name
  consistency for virtual-node markers, and read-only exposure of public
  feature metadata.
- `Validation Evidence`: Pass. Planned evidence includes `pytest` coverage for
  root/subpackage exports, import behavior without optional tree I/O, metadata
  naming, and preserved core user workflows.
- `Reproducibility`: Pass. No workflow engine is added. Reproducibility is
  maintained through deterministic export lists, documentation alignment, and
  small package-level import tests.
- `Pragmatism`: Pass with one explicit deferral: no broad compatibility shim or
  deprecation framework will be introduced unless tests show it is needed.
  This keeps the change small while preserving the documented primary workflow.

## Project Structure

### Documentation (this feature)

```text
specs/[###-feature]/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)
<!--
  ACTION REQUIRED: Replace the placeholder tree below with the concrete layout
  for this feature. Delete unused options and expand the chosen structure with
  real paths (e.g., apps/admin, packages/something). The delivered plan must
  not include Option labels.
-->

```text
src/
└── phylognn/
    ├── __init__.py
    ├── data/
    │   ├── __init__.py
    │   ├── feature_engineer.py
    │   ├── converter.py
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
├── test_feature_engineer.py
└── test_data_conversion.py

specs/001-api-exposure-refactor/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
└── contracts/
    └── public-api-surface.md
```

**Structure Decision**: Keep the existing single-package layout under
`src/phylognn/`. This refactor is constrained to package export boundaries,
metadata contracts, and aligned tests/docs rather than introducing new runtime
framework layers or service structures.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| None | N/A | The feature can be delivered within existing package boundaries |
