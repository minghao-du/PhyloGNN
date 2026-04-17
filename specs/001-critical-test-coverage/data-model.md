# Data Model: Critical Test Coverage

## Entity: CoverageTier

- **Purpose**: Defines the minimum rigor expected for a group of in-scope modules.
- **Fields**:
  - `name`: Canonical tier identifier.
  - `scope_patterns`: Package modules covered by the tier.
  - `risk_level`: Relative regression risk that justifies stricter or lighter expectations.
  - `minimum_expectation`: Human-readable definition of what “complete enough” means for the tier.
  - `required_evidence`: Test categories or artifacts that must exist for the tier.
- **Relationships**:
  - One `CoverageTier` applies to many `CoverageTarget` records.

## Entity: CoverageTarget

- **Purpose**: Represents an in-scope public module or externally visible API surface under `src/phylognn/`.
- **Fields**:
  - `canonical_name`: Module path or public API surface name.
  - `target_type`: Public module, facade export surface, or optional-boundary module.
  - `package_area`: `data`, `models`, `training`, `io`, `utils`, or top-level package facade.
  - `tier_name`: Associated `CoverageTier`.
  - `contracts_protected`: Scientific/data or public API contracts that tests must assert.
  - `failure_modes`: Invalid-input or environment failures that must be covered.
  - `status`: Planned, covered, or exception.
- **Relationships**:
  - One `CoverageTarget` has many `CoverageScenario` records.
  - One `CoverageTarget` may have zero or one `ExceptionRecord`.

## Entity: CoverageScenario

- **Purpose**: Describes one observable behavior that a test must protect.
- **Fields**:
  - `scenario_name`: Short identifier.
  - `path_type`: Success, failure, edge, or regression.
  - `input_shape`: Tree, graph, tensor, configuration, or import-surface setup needed to exercise the behavior.
  - `expected_outcome`: Observable result, exported symbol, graph field, raised error class, or deterministic property.
  - `test_location`: Pytest module or test case responsible for the check.
- **Relationships**:
  - Many `CoverageScenario` records belong to one `CoverageTarget`.

## Entity: CoverageMappingEntry

- **Purpose**: The record format used in the human-readable coverage mapping artifact.
- **Fields**:
  - `target`: Reference to the `CoverageTarget`.
  - `test_files`: Pytest files that implement coverage for the target.
  - `scenario_summary`: Summary of success-path, failure-path, and regression evidence.
  - `notes`: Optional explanation for special setup or shared fixtures.
  - `exception_ref`: Optional link to an `ExceptionRecord`.

## Entity: ExceptionRecord

- **Purpose**: Documents a rare in-scope behavior that cannot yet be fully automated.
- **Fields**:
  - `target`: The blocked `CoverageTarget`.
  - `blocked_behavior`: Specific uncovered contract.
  - `reason`: Why automation is currently blocked.
  - `impact`: What confidence remains at risk.
  - `manual_verification`: Exact manual check still required.
  - `follow_up_plan`: Next step required to remove the exception later.
- **Relationships**:
  - One `ExceptionRecord` belongs to one `CoverageTarget`.

## Proposed Tier Shapes

### Tier 1: Core Scientific Logic

- **Scope**: `phylognn.data.*`, `phylognn.models.*`, `phylognn.training.*`
- **Expectation**: Strong success-path, failure-path, and regression coverage using realistic fixtures and direct assertions on scientific/data contracts.

### Tier 2: Public Facade And Boundary Modules

- **Scope**: `phylognn.__init__`, `phylognn.data.__init__`, `phylognn.models.__init__`, `phylognn.training.__init__`, `phylognn.io`, and other externally visible facade surfaces
- **Expectation**: Strong public-contract coverage with lighter internal branching expectations, emphasizing export names, lazy resolution behavior, and optional dependency boundaries.
