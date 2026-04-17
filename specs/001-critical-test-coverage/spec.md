# Feature Specification: Critical Test Coverage

**Feature Branch**: `001-critical-test-coverage`  
**Created**: 2026-04-16  
**Status**: Draft  
**Input**: User description: "所有的关键代码,都要有完整的test,@tests文件要补充完整"

## Clarifications

### Session 2026-04-16

- Q: Which code scope must receive complete test coverage for this feature? → A: Cover public modules and externally visible APIs under `src/phylognn/`; `examples/` are not part of the complete coverage target.
- Q: How should minimum test coverage expectations be defined across the in-scope codebase? → A: Use tiered thresholds, with stricter expectations for core data, model, and training modules than for public facade modules.
- Q: What evidence is required to prove the coverage work is complete? → A: Maintain a coverage mapping checklist that links each in-scope module or public API to the automated tests that protect it.
- Q: How should exceptions be handled when some in-scope behaviors cannot be fully automated? → A: Allow only a small number of manual exceptions, and require each one to be explicitly documented with its reason, impact, and follow-up plan.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Protect Critical Package Behavior (Priority: P1)

As a package maintainer, I need automated tests for every critical code path so that regressions in tree conversion, feature engineering, model behavior, and training utilities are detected before release.

**Why this priority**: Regressions in core scientific and graph-building logic can silently corrupt downstream experiments, making this the highest-risk area to leave untested.

**Independent Test**: This story is complete when a maintainer can run the targeted automated tests for the identified critical modules and see failures whenever a protected contract is intentionally broken.

**Acceptance Scenarios**:

1. **Given** a critical public function or class in the package, **When** its expected success path is exercised, **Then** an automated test verifies the documented output, metadata, and observable side effects.
2. **Given** invalid tree data, tensor data, configuration, or other required inputs for a critical code path, **When** the behavior is exercised, **Then** an automated test verifies the documented fast-fail error and message category.

---

### User Story 2 - Make Test Coverage Actionable For Contributors (Priority: P2)

As a contributor, I need the `tests/` directory to clearly map critical modules to focused test files or sections so that I can safely change code without guessing which behaviors must stay stable.

**Why this priority**: Good coverage loses value if maintainers cannot tell which tests protect which contracts or where new regression cases belong.

**Independent Test**: This story is complete when a contributor can identify the responsible automated tests for each critical module and extend them without reverse-engineering the entire suite.

**Acceptance Scenarios**:

1. **Given** a critical module in `src/phylognn/`, **When** a contributor inspects `tests/`, **Then** there is a dedicated or clearly grouped test location covering that module's primary contracts.
2. **Given** an existing bug fix or refactor in a critical module, **When** a contributor adds or updates a regression test, **Then** the new case fits the existing test organization without creating ambiguous ownership.

---

### User Story 3 - Preserve Public And Scientific Contracts Across Releases (Priority: P3)

As a release maintainer, I need tests that protect public API exposure and scientific data contracts so that packaging changes, refactors, or optional dependency differences do not change observable behavior unexpectedly.

**Why this priority**: Release confidence depends on stable public exports and reproducible graph/data contracts, even when internal implementations evolve.

**Independent Test**: This story is complete when release validation can rely on automated tests to confirm that the exposed APIs and scientific data contracts still behave as documented.

**Acceptance Scenarios**:

1. **Given** a supported public import path or package facade, **When** release validation runs, **Then** automated tests confirm that the expected symbols remain available and behave consistently.
2. **Given** a representative phylogenetic input and configuration, **When** conversion or training-related workflows run under test, **Then** automated checks confirm stable graph fields, validation behavior, and reproducible ordering guarantees.

### Edge Cases

- What happens when optional dependencies required for a supported workflow are missing at runtime?
- How does the system handle malformed tree inputs, empty inputs, single-node trees, or inconsistent node metadata?
- How does the system handle tensor shape, dtype, or dimensionality mismatches in model and training code paths?
- What happens when a critical public API is re-exported incorrectly or removed from a package facade?
- How are deterministic ordering and graph-level metadata verified when the same logical input is processed repeatedly?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST define and document the set of critical code paths that require direct automated test coverage before this feature is considered complete.
- **FR-002**: The system MUST provide at least one automated success-path test for every identified critical public API, workflow entry point, or validator-backed behavior.
- **FR-003**: The system MUST provide automated failure-path or edge-case coverage for every identified critical code path where invalid inputs, inconsistent state, or missing prerequisites are expected to fail fast.
- **FR-004**: The `tests/` directory MUST be expanded or reorganized so that every identified critical module is covered by a dedicated test file or an explicitly named grouped section with clear ownership.
- **FR-005**: Automated tests MUST verify observable scientific and data-contract behavior, including required graph fields, tensor expectations, metadata preservation, ordering guarantees, and documented validation rules where applicable.
- **FR-006**: Automated tests MUST verify that public package exports and facade modules continue to expose the documented symbols and behavior expected by downstream users.
- **FR-007**: Automated tests MUST prefer `src/phylognn/` behavior and current package contracts as the source of truth whenever example scripts or legacy usage patterns differ.
- **FR-008**: Any critical behavior that cannot yet be covered automatically MUST be explicitly identified in the validation plan together with the reason it remains a manual verification gap.
- **FR-009**: Complete coverage for this feature MUST apply to public modules and externally visible APIs under `src/phylognn/`; example scripts are excluded from the complete coverage target and may only receive targeted regression tests when they reveal package-level contract gaps.
- **FR-010**: Coverage expectations MUST be defined in tiers, with stricter thresholds for core data, model, and training modules than for public facade or re-export modules.
- **FR-011**: The feature MUST maintain an explicit coverage mapping artifact that links each in-scope module or public API to the automated tests that verify its success paths, failure modes, and regressions.
- **FR-012**: Any in-scope behavior that cannot be fully automated MUST remain an exception case only when it is explicitly documented with the reason, impact, and intended follow-up, and such exceptions MUST stay limited in number.

### Key Entities *(include if feature involves data)*

- **Critical Code Path**: A public API, core transformation, validator, model behavior, or training workflow whose failure would materially affect package correctness, reproducibility, or downstream usage.
- **Coverage Mapping**: The explicit relationship between a critical code path and the automated tests that protect its success cases, edge cases, and regressions.
- **Regression Fixture**: A representative tree, graph, tensor, or configuration input used to verify that previously observed or likely failures remain protected by automated tests.

## Scientific And Data Contracts *(mandatory for package, model, graph, or workflow changes)*

- **Input Contracts**: Tests must exercise supported phylogenetic tree inputs, graph conversion inputs, tensor-bearing model inputs, and training or metric configurations using valid and invalid values that reflect current package contracts.
- **Output Contracts**: Tests must confirm that critical workflows produce the documented graphs, tensors, metrics, metadata fields, and public return values expected by downstream users, including stable names and required attributes.
- **Failure Modes**: Tests must verify that malformed trees, incompatible tensor shapes or dtypes, missing required attributes, and unsupported optional-dependency paths fail explicitly rather than silently degrading behavior.
- **Reproducibility Notes**: Tests must protect deterministic ordering, stable graph metadata, and repeatable observable outcomes for representative inputs wherever the package currently promises reproducible behavior.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of the critical code paths identified for this feature have at least one mapped automated test location in `tests/`.
- **SC-002**: 100% of identified critical public APIs have both a success-path verification and at least one failure-path or edge-case verification, unless explicitly documented as a justified manual gap.
- **SC-003**: Maintainers can run the targeted validation suite for this feature and observe all new or updated tests passing in a clean local run before merge.
- **SC-004**: Intentionally breaking at least one representative protected contract in each covered critical area causes one or more automated tests to fail, demonstrating that the new coverage is regression-detecting rather than purely smoke-level.
- **SC-005**: The feature defines measurable coverage tiers so maintainers can determine, before merge, whether core data/model/training modules and public facade modules each meet their respective minimum target.
- **SC-006**: Maintainers can inspect a single coverage mapping artifact and determine, without reading source code line by line, which automated tests protect every in-scope module and public API.
- **SC-007**: Any manual exception retained at merge time is individually documented, justified, and small enough in count that maintainers can review the full exception list during release validation.

## Assumptions

- Critical code primarily includes public package APIs and high-risk scientific workflows in data conversion, feature engineering, model behavior, training utilities, and package export surfaces.
- The complete coverage target is limited to public modules and externally visible APIs in `src/phylognn/`, not to example scripts.
- Core scientific and training modules carry higher regression risk than facade or re-export modules and therefore justify stricter minimum coverage expectations.
- Existing package behavior, docstrings, and current tests are the baseline source of truth unless they directly conflict with `src/phylognn/` contracts.
- The feature focuses on closing meaningful automated test gaps rather than changing the scientific behavior of the package itself.
- Example scripts may remain less strict than package contracts, and any disagreement is resolved in favor of `src/phylognn/` and the maintained tests.

## Validation Plan

- **Unit/Regression Tests**: Expand or add focused pytest coverage for `data`, `models`, `training`, package facade, and validation-heavy behaviors so each critical module has direct success, failure, and regression assertions.
- **Unit/Regression Tests**: Expand or add focused pytest coverage for `data`, `models`, `training`, package facade, and validation-heavy behaviors so each critical module has direct success, failure, and regression assertions, and record those links in a coverage mapping artifact.
- **Integration/Workflow Tests**: Run representative end-to-end checks that cover tree-to-graph conversion, public import surfaces, and at least one model or training workflow path using stable fixtures.
- **Manual Verification**: Only retain manual verification for critical behaviors that cannot be automated without unsupported external dependencies or non-deterministic setup, and document each such gap explicitly before merge.
- **Manual Verification**: Only retain a small number of manual exceptions for critical behaviors that cannot be automated without unsupported external dependencies or non-deterministic setup, and document each exception with its reason, impact, and follow-up plan before merge.
