---

description: "Task list for API Exposure Refactor"
---

# Tasks: API Exposure Refactor

**Input**: Design documents from `/specs/001-api-exposure-refactor/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Include validation tasks for every non-trivial behavioral change.
Tests may be omitted only for documentation-only or pure maintenance work, and
the omission must be justified in the plan.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/`, `tests/` at repository root
- Paths shown below assume the existing PhyloGNN package layout under `src/phylognn/`

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Prepare the repo for focused API-surface refactoring and validation

- [X] T001 Review and align the feature artifacts in `specs/001-api-exposure-refactor/plan.md`, `research.md`, `data-model.md`, and `contracts/public-api-surface.md`
- [X] T002 Create `tests/test_public_api.py` for root and subpackage import-surface validation
- [X] T003 [P] Create `tests/test_training_api.py` for curated training export and metadata regression coverage
- [X] T004 [P] Capture stable import examples and expected public names in `specs/001-api-exposure-refactor/quickstart.md`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Establish the shared public-contract baseline before story-specific changes

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T005 Define the canonical stable, advanced, and internal API boundaries in `src/phylognn/__init__.py`, `src/phylognn/data/__init__.py`, `src/phylognn/models/__init__.py`, and `src/phylognn/training/__init__.py`
- [X] T006 [P] Add fail-fast regression assertions for export lists, nonexistent names, and metadata naming in `tests/test_public_api.py`
- [X] T007 [P] Add regression coverage for feature metadata immutability and virtual-node naming consistency in `tests/test_feature_engineer.py` and `tests/test_data_conversion.py`
- [X] T008 Document the shared public-contract expectations in `specs/001-api-exposure-refactor/contracts/public-api-surface.md`

**Checkpoint**: Foundational API contract and validation scaffolding are ready

---

## Phase 3: User Story 1 - Discover Stable Entry Points (Priority: P1) 🎯 MVP

**Goal**: Deliver a small, dependable top-level PhyloGNN import surface for the main workflow

**Independent Test**: A user can import the documented stable root-package names from `phylognn` and run the core feature-engineering plus graph-conversion workflow without needing internal module knowledge.

### Tests for User Story 1 ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T009 [P] [US1] Add root-package import contract tests in `tests/test_public_api.py`
- [X] T010 [P] [US1] Add regression tests for `__version__` exposure in `tests/test_public_api.py`
- [X] T011 [P] [US1] Add workflow import coverage for the stable top-level preprocessing path in `tests/test_data_conversion.py`

### Implementation for User Story 1

- [X] T012 [US1] Curate the stable root-package exports and rename `version` to `__version__` in `src/phylognn/__init__.py`
- [X] T013 [US1] Align top-level package docstrings and import examples in `src/phylognn/__init__.py`
- [X] T014 [US1] Update stable user-facing import examples in `specs/001-api-exposure-refactor/quickstart.md`
- [X] T015 [US1] Update the public root-layer contract in `specs/001-api-exposure-refactor/contracts/public-api-surface.md`

**Checkpoint**: User Story 1 should provide a clear and testable stable top-level API

---

## Phase 4: User Story 2 - Use Advanced APIs Intentionally (Priority: P2)

**Goal**: Expose curated advanced subpackage APIs without promoting accidental low-level exports

**Independent Test**: An advanced user can import the documented advanced names from `phylognn.data`, `phylognn.models`, and `phylognn.training`, while low-level implementation symbols are no longer represented as package-level public API.

### Tests for User Story 2 ⚠️

- [X] T016 [P] [US2] Add curated subpackage import tests for `phylognn.data`, `phylognn.models`, and `phylognn.training` in `tests/test_public_api.py`
- [X] T017 [P] [US2] Add training export regression tests for datasets, metrics, and `create_default_trainer` in `tests/test_training_api.py`
- [X] T018 [P] [US2] Add regression tests that low-level model-layer symbols are not exported from `src/phylognn/models/__init__.py` in `tests/test_public_api.py`

### Implementation for User Story 2

- [X] T019 [US2] Curate advanced data exports and remove default tree I/O promotion from `src/phylognn/data/__init__.py`
- [X] T020 [US2] Curate advanced model exports in `src/phylognn/models/__init__.py`
- [X] T021 [US2] Fix training package exports, `__all__`, and intended helpers in `src/phylognn/training/__init__.py`
- [X] T022 [US2] Align subpackage import examples and supported advanced API docs in `specs/001-api-exposure-refactor/quickstart.md`
- [X] T023 [US2] Update the advanced-layer contract definitions in `specs/001-api-exposure-refactor/contracts/public-api-surface.md`

**Checkpoint**: User Stories 1 and 2 should now both work independently

---

## Phase 5: User Story 3 - Refactor Internals Safely (Priority: P3)

**Goal**: Reduce coupling to internal building blocks and optional subsystems so maintainers can refactor safely

**Independent Test**: Internal layer classes and optional tree I/O no longer define the stable public contract, while the documented core workflow continues to work without optional tree-loading dependencies.

### Tests for User Story 3 ⚠️

- [X] T024 [P] [US3] Add import-behavior regression tests for root usage without optional tree I/O coupling in `tests/test_public_api.py`
- [X] T025 [P] [US3] Add regression tests for read-only `feature_names` and `available_features` exposure in `tests/test_feature_engineer.py`
- [X] T026 [P] [US3] Add regression tests for canonical virtual-node feature naming in `tests/test_data_conversion.py`

### Implementation for User Story 3

- [X] T027 [US3] Expose read-only public feature metadata from `src/phylognn/data/feature_engineer.py`
- [X] T028 [US3] Standardize canonical virtual-node feature naming in `src/phylognn/data/converter.py`
- [X] T029 [US3] Decouple optional tree I/O from the default root import path in `src/phylognn/__init__.py` and `src/phylognn/data/__init__.py`
- [X] T030 [US3] Update package and module docstrings to distinguish internal versus public APIs in `src/phylognn/__init__.py`, `src/phylognn/data/__init__.py`, `src/phylognn/models/__init__.py`, and `src/phylognn/training/__init__.py`
- [X] T031 [US3] Refresh maintainer-facing API-boundary notes in `specs/001-api-exposure-refactor/research.md` and `data-model.md`

**Checkpoint**: All user stories should now be independently functional

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final validation and cleanup across the full feature

- [X] T032 [P] Run focused API and regression tests in `tests/test_public_api.py`, `tests/test_training_api.py`, `tests/test_feature_engineer.py`, and `tests/test_data_conversion.py`
- [X] T033 Run formatting and lint checks on touched files with `black --check src tests` and `ruff check src tests`
- [X] T034 [P] Update or add user-facing import examples in `examples/` if current examples conflict with the curated API surface
- [X] T035 Perform a final manual quickstart validation against `specs/001-api-exposure-refactor/quickstart.md`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies, start immediately
- **Foundational (Phase 2)**: Depends on Setup completion and blocks all user stories
- **User Story 1 (Phase 3)**: Depends on Foundational completion
- **User Story 2 (Phase 4)**: Depends on Foundational completion and should build on the stable root API from User Story 1
- **User Story 3 (Phase 5)**: Depends on Foundational completion and should integrate with the curated surfaces completed in User Stories 1 and 2
- **Polish (Phase 6)**: Depends on completion of all desired user stories

### User Story Dependencies

- **User Story 1 (P1)**: MVP and first delivery target
- **User Story 2 (P2)**: Depends conceptually on the stable top-level API but remains independently testable once foundational work is complete
- **User Story 3 (P3)**: Depends on the intended public/export boundaries established by User Stories 1 and 2

### Within Each User Story

- Tests MUST be written and FAIL before implementation for non-trivial changes
- Export contract tests before `__init__.py` refactors
- Package boundary changes before documentation alignment
- Story-specific validation before moving to the next story

### Parallel Opportunities

- `T003` and `T004` can run in parallel after `T002`
- `T006` and `T007` can run in parallel after `T005`
- `T009`, `T010`, and `T011` can run in parallel within User Story 1
- `T016`, `T017`, and `T018` can run in parallel within User Story 2
- `T024`, `T025`, and `T026` can run in parallel within User Story 3
- `T032` and `T034` can run in parallel during Polish

---

## Parallel Example: User Story 1

```bash
# Launch all User Story 1 tests together:
Task: "Add root-package import contract tests in tests/test_public_api.py"
Task: "Add regression tests for __version__ exposure in tests/test_public_api.py"
Task: "Add workflow import coverage for the stable top-level preprocessing path in tests/test_data_conversion.py"
```

---

## Parallel Example: User Story 2

```bash
# Launch all User Story 2 tests together:
Task: "Add curated subpackage import tests for phylognn.data, phylognn.models, and phylognn.training in tests/test_public_api.py"
Task: "Add training export regression tests for datasets, metrics, and create_default_trainer in tests/test_training_api.py"
Task: "Add regression tests that low-level model-layer symbols are not exported from src/phylognn/models/__init__.py in tests/test_public_api.py"
```

---

## Parallel Example: User Story 3

```bash
# Launch all User Story 3 tests together:
Task: "Add import-behavior regression tests for root usage without optional tree I/O coupling in tests/test_public_api.py"
Task: "Add regression tests for read-only feature_names and available_features exposure in tests/test_feature_engineer.py"
Task: "Add regression tests for canonical virtual-node feature naming in tests/test_data_conversion.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: User Story 1
4. Validate the stable root-package workflow
5. Stop for review if a minimal API cleanup release is desired

### Incremental Delivery

1. Deliver User Story 1 for the stable top-level API
2. Add User Story 2 for curated advanced subpackage APIs
3. Add User Story 3 for safe internal boundary tightening and optional tree I/O decoupling
4. Finish with cross-cutting validation and documentation cleanup

### Parallel Team Strategy

1. One contributor handles shared test scaffolding in `tests/test_public_api.py`
2. One contributor handles package export curation across `src/phylognn/*/__init__.py`
3. One contributor handles data-contract updates in `feature_engineer.py` and `converter.py`
4. Merge on story boundaries with regression tests green before advancing

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] labels map tasks directly to spec user stories
- Each user story remains independently testable with explicit import-contract criteria
- All tasks include concrete file paths and are immediately actionable
- Suggested MVP scope: Phase 3 / User Story 1 only
