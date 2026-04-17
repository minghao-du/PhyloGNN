# Research: Critical Test Coverage

## Decision 1: Use risk-tiered coverage expectations instead of one global threshold

- **Decision**: Define stricter minimum coverage expectations for `phylognn.data`, `phylognn.models`, and `phylognn.training` than for facade modules such as package `__init__` surfaces and `phylognn.io`.
- **Rationale**: Core scientific and training logic carries higher regression risk than lazy export facades. A tiered model matches the clarified spec and prevents low-risk facade files from diluting expectations for high-risk scientific code.
- **Alternatives considered**:
  - One global threshold for all modules: rejected because it hides risk differences.
  - No threshold, only behavioral tests: rejected because completion becomes subjective.

## Decision 2: Maintain a human-readable coverage mapping artifact alongside tests

- **Decision**: Use a Markdown coverage-mapping artifact that links each in-scope public module or API surface to the pytest modules and scenarios that protect it.
- **Rationale**: The feature must prove completeness, not only add tests. A readable mapping lets maintainers verify scope coverage quickly and makes future test additions easier to place correctly.
- **Alternatives considered**:
  - Rely only on line coverage output: rejected because it does not show which public contracts are protected.
  - Encode the mapping only in test names: rejected because discoverability remains weak.

## Decision 3: Organize new tests by package surface, not by broad workflow only

- **Decision**: Expand `tests/` so package areas map clearly to test ownership: top-level/public API tests, data tests, model tests, training tests, and optional I/O boundary tests.
- **Rationale**: The clarified scope is “all public modules and externally visible APIs under `src/phylognn/`”. Grouping tests by package surface makes missing coverage visible and supports targeted iteration.
- **Alternatives considered**:
  - Keep only the current small set of broad test files: rejected because ownership becomes ambiguous as coverage grows.
  - Build a single monolithic regression file: rejected because maintenance cost grows too quickly.

## Decision 4: Allow only bounded, explicit manual exceptions

- **Decision**: Permit only a small number of manual exceptions, each documented with reason, impact, and follow-up plan when full automation is blocked by optional dependencies or unstable environment requirements.
- **Rationale**: Some tree I/O or scientific runtime paths may depend on extras that are not always available. A bounded exception policy preserves rigor without blocking all progress.
- **Alternatives considered**:
  - Require full automation for every path immediately: rejected because optional-dependency boundaries could block delivery.
  - Allow untracked exceptions: rejected because coverage completeness would become unverifiable.

## Decision 5: Treat curated export surfaces as first-class contracts

- **Decision**: Keep package and subpackage `__all__` surfaces, lazy export behavior, and optional I/O boundaries as explicit contract targets in the test plan.
- **Rationale**: Existing tests already protect curated exports, and the spec requires public API stability. These surfaces are part of the user-facing contract even when their implementations are simple.
- **Alternatives considered**:
  - Focus only on heavy scientific modules: rejected because export regressions still break users.
