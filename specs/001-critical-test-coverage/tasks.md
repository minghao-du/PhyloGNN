# Tasks: Critical Test Coverage

**Input**: Design documents from `/specs/001-critical-test-coverage/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: Include validation tasks for every non-trivial behavioral change. This feature is test-centric, so each user story begins with test work and ends with explicit validation or coverage-mapping updates.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Establish the coverage-tracking artifacts and scope inventory used by all later work

- [X] T001 Create the coverage mapping scaffold in /Users/Minghao/Research/PhyloGNN/specs/001-critical-test-coverage/coverage-mapping.md
- [X] T002 Create the bounded exception tracker in /Users/Minghao/Research/PhyloGNN/specs/001-critical-test-coverage/manual-exceptions.md
- [X] T003 Record every in-scope public module and externally visible API target in /Users/Minghao/Research/PhyloGNN/specs/001-critical-test-coverage/coverage-mapping.md
- [X] T004 Record current test ownership gaps against `src/phylognn/` in /Users/Minghao/Research/PhyloGNN/specs/001-critical-test-coverage/coverage-mapping.md

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Build the shared fixtures and validation scaffolding that every user story depends on

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T005 Create shared scientific test helpers in /Users/Minghao/Research/PhyloGNN/tests/support.py
- [X] T006 Update reusable tree, tensor, and optional-dependency fixtures in /Users/Minghao/Research/PhyloGNN/tests/conftest.py
- [X] T007 Define tier assignments, acceptance rules, and validation command placeholders in /Users/Minghao/Research/PhyloGNN/specs/001-critical-test-coverage/coverage-mapping.md
- [X] T008 Capture the default manual-exception template with reason, impact, and follow-up fields in /Users/Minghao/Research/PhyloGNN/specs/001-critical-test-coverage/manual-exceptions.md

**Checkpoint**: Shared fixtures, coverage tiers, and exception mechanics are ready; user story work can proceed

---

## Phase 3: User Story 1 - Protect Critical Package Behavior (Priority: P1) 🎯 MVP

**Goal**: Add strong success-path, failure-path, and regression coverage for the core scientific and training modules under `phylognn.data`, `phylognn.models`, and `phylognn.training`

**Independent Test**: Run the targeted core-module pytest files and confirm they fail when representative scientific contracts or fail-fast validations are intentionally broken

### Tests for User Story 1 ⚠️

- [X] T009 [P] [US1] Expand feature engineering success/failure coverage in /Users/Minghao/Research/PhyloGNN/tests/test_feature_engineer.py
- [X] T010 [P] [US1] Expand tree conversion and graph metadata regression coverage in /Users/Minghao/Research/PhyloGNN/tests/test_data_conversion.py
- [X] T011 [P] [US1] Add model contract and validation tests in /Users/Minghao/Research/PhyloGNN/tests/test_model_contracts.py
- [X] T012 [P] [US1] Add trainer, config, dataset, and metric contract tests in /Users/Minghao/Research/PhyloGNN/tests/test_training_components.py

### Implementation for User Story 1

- [X] T013 [US1] Tighten or clarify feature-engineering contract docstrings and fail-fast validation in /Users/Minghao/Research/PhyloGNN/src/phylognn/data/feature_engineer.py
- [X] T014 [US1] Tighten or clarify graph-conversion contract docstrings and validation in /Users/Minghao/Research/PhyloGNN/src/phylognn/data/converter.py
- [X] T015 [US1] Tighten or clarify core model and trainer validation surfaces in /Users/Minghao/Research/PhyloGNN/src/phylognn/models/base.py and /Users/Minghao/Research/PhyloGNN/src/phylognn/training/trainer.py
- [X] T016 [US1] Record US1 target-to-test ownership and scenario summaries in /Users/Minghao/Research/PhyloGNN/specs/001-critical-test-coverage/coverage-mapping.md
- [X] T017 [US1] Record any remaining core-module automation gaps in /Users/Minghao/Research/PhyloGNN/specs/001-critical-test-coverage/manual-exceptions.md

**Checkpoint**: Core data, model, and training behavior is covered and independently testable

---

## Phase 4: User Story 2 - Make Test Coverage Actionable For Contributors (Priority: P2)

**Goal**: Reorganize and document tests so contributors can see ownership and extend coverage without reverse-engineering the suite

**Independent Test**: A contributor can inspect `tests/` and the coverage map, find the responsible test location for each in-scope public module, and add a regression in the correct file without guessing

### Tests for User Story 2 ⚠️

- [X] T018 [P] [US2] Add dedicated optional I/O boundary tests in /Users/Minghao/Research/PhyloGNN/tests/test_io_api.py
- [X] T019 [P] [US2] Add dedicated facade and utility surface tests in /Users/Minghao/Research/PhyloGNN/tests/test_models_api.py and /Users/Minghao/Research/PhyloGNN/tests/test_utils_api.py
- [X] T020 [P] [US2] Refine top-level and training export coverage in /Users/Minghao/Research/PhyloGNN/tests/test_public_api.py and /Users/Minghao/Research/PhyloGNN/tests/test_training_api.py

### Implementation for User Story 2

- [X] T021 [US2] Reorganize public-surface assertions so each in-scope facade has a dedicated ownership location in /Users/Minghao/Research/PhyloGNN/tests/test_public_api.py, /Users/Minghao/Research/PhyloGNN/tests/test_models_api.py, /Users/Minghao/Research/PhyloGNN/tests/test_training_api.py, /Users/Minghao/Research/PhyloGNN/tests/test_io_api.py, and /Users/Minghao/Research/PhyloGNN/tests/test_utils_api.py
- [X] T022 [US2] Update the coverage mapping entries with final contributor-facing test locations in /Users/Minghao/Research/PhyloGNN/specs/001-critical-test-coverage/coverage-mapping.md
- [X] T023 [US2] Update contributor verification and extension guidance in /Users/Minghao/Research/PhyloGNN/specs/001-critical-test-coverage/quickstart.md

**Checkpoint**: Test ownership is discoverable and every in-scope facade has an obvious home in `tests/`

---

## Phase 5: User Story 3 - Preserve Public And Scientific Contracts Across Releases (Priority: P3)

**Goal**: Protect release-time public exports, optional dependency boundaries, and deterministic scientific/data contracts with regression-focused validation

**Independent Test**: Run the release-facing contract tests and confirm public exports, optional boundaries, and deterministic scientific outputs remain stable across repeated execution

### Tests for User Story 3 ⚠️

- [X] T024 [P] [US3] Add lazy-export and public-surface regression checks in /Users/Minghao/Research/PhyloGNN/tests/test_public_api.py and /Users/Minghao/Research/PhyloGNN/tests/test_io_api.py
- [X] T025 [P] [US3] Add deterministic graph, metadata, and validation regression tests in /Users/Minghao/Research/PhyloGNN/tests/test_data_contracts.py
- [X] T026 [P] [US3] Add release-level import and contract smoke tests in /Users/Minghao/Research/PhyloGNN/tests/test_release_contracts.py

### Implementation for User Story 3

- [X] T027 [US3] Update curated export surfaces and optional-boundary behavior as required by failing release-contract tests in /Users/Minghao/Research/PhyloGNN/src/phylognn/__init__.py, /Users/Minghao/Research/PhyloGNN/src/phylognn/data/__init__.py, /Users/Minghao/Research/PhyloGNN/src/phylognn/models/__init__.py, /Users/Minghao/Research/PhyloGNN/src/phylognn/training/__init__.py, and /Users/Minghao/Research/PhyloGNN/src/phylognn/io.py
- [X] T028 [US3] Update release-validation entries and exception references in /Users/Minghao/Research/PhyloGNN/specs/001-critical-test-coverage/coverage-mapping.md and /Users/Minghao/Research/PhyloGNN/specs/001-critical-test-coverage/manual-exceptions.md

**Checkpoint**: Public API and scientific contracts are protected for release validation

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final validation and cleanup across all user stories

- [X] T029 [P] Run the focused validation commands and refresh execution guidance in /Users/Minghao/Research/PhyloGNN/specs/001-critical-test-coverage/quickstart.md
- [X] T030 [P] Verify every in-scope target appears exactly once with real test locations in /Users/Minghao/Research/PhyloGNN/specs/001-critical-test-coverage/coverage-mapping.md
- [X] T031 Run the full pytest suite and record any final bounded exceptions in /Users/Minghao/Research/PhyloGNN/specs/001-critical-test-coverage/manual-exceptions.md

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies; starts immediately
- **Foundational (Phase 2)**: Depends on Setup completion and blocks all story work
- **User Story 1 (Phase 3)**: Depends on Foundational completion; defines the MVP
- **User Story 2 (Phase 4)**: Depends on Foundational completion and may use US1 test helpers, but remains independently testable
- **User Story 3 (Phase 5)**: Depends on Foundational completion and benefits from earlier test organization, but remains independently testable
- **Polish (Phase 6)**: Depends on all desired story phases being complete

### User Story Dependencies

- **US1 (P1)**: No dependency on other user stories
- **US2 (P2)**: No hard dependency on US1, but it can reuse any shared helpers added in US1
- **US3 (P3)**: No hard dependency on US1 or US2, but it integrates best after core and facade coverage exists

### Within Each User Story

- Tests must be written or expanded before source adjustments for that story
- Shared contract or validation fixes follow the failing tests they unblock
- Coverage mapping and exception records are updated after the story’s tests and code changes are in place
- Story-specific validation must pass before moving to the next priority

### Parallel Opportunities

- T009-T012 can run in parallel because they target different test files
- T018-T020 can run in parallel because they target different facade-oriented test files
- T024-T026 can run in parallel because they target different release-regression test files
- T029-T030 can run in parallel because they update different documentation artifacts

---

## Parallel Example: User Story 1

```bash
# Launch core-module test work together:
Task: "Expand feature engineering success/failure coverage in tests/test_feature_engineer.py"
Task: "Expand tree conversion and graph metadata regression coverage in tests/test_data_conversion.py"
Task: "Add model contract and validation tests in tests/test_model_contracts.py"
Task: "Add trainer, config, dataset, and metric contract tests in tests/test_training_components.py"
```

## Parallel Example: User Story 2

```bash
# Launch facade-oriented test work together:
Task: "Add dedicated optional I/O boundary tests in tests/test_io_api.py"
Task: "Add dedicated facade and utility surface tests in tests/test_models_api.py and tests/test_utils_api.py"
Task: "Refine top-level and training export coverage in tests/test_public_api.py and tests/test_training_api.py"
```

## Parallel Example: User Story 3

```bash
# Launch release-regression work together:
Task: "Add lazy-export and public-surface regression checks in tests/test_public_api.py and tests/test_io_api.py"
Task: "Add deterministic graph, metadata, and validation regression tests in tests/test_data_contracts.py"
Task: "Add release-level import and contract smoke tests in tests/test_release_contracts.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: User Story 1
4. Stop and validate the targeted core scientific coverage before proceeding

### Incremental Delivery

1. Finish Setup + Foundational to establish shared fixtures, mapping, and exception handling
2. Deliver US1 for core scientific correctness coverage
3. Deliver US2 for contributor-facing organization and discoverability
4. Deliver US3 for release-facing public and scientific contract protection
5. Finish with polish validation and exception review

### Parallel Team Strategy

1. One contributor prepares shared fixtures and coverage artifacts in Phases 1-2
2. After Foundational completion:
   - Contributor A handles US1 core scientific coverage
   - Contributor B handles US2 facade/test organization
   - Contributor C handles US3 release-regression coverage
3. Merge story branches only after each story’s independent test criterion passes

---

## Notes

- [P] tasks touch different files and can be executed in parallel safely
- [US1], [US2], and [US3] labels preserve traceability back to the specification
- `coverage-mapping.md` and `manual-exceptions.md` are required deliverables for proving completeness
- Prefer realistic trees, tensors, and package imports over mocks when extending tests
