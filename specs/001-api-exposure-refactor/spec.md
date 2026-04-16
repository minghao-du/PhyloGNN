# Feature Specification: API Exposure Refactor

**Feature Branch**: `001-api-exposure-refactor`  
**Created**: 2026-04-16  
**Status**: Draft  
**Input**: User description: "按照@doc/api_exposure_refactor_plan.md，写Specification"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Discover Stable Entry Points (Priority: P1)

As a PhyloGNN user, I want the package to expose a small, clear set of stable
entry points so I can start feature engineering, graph conversion, training,
and model use without guessing which imports are official.

**Why this priority**: This is the main user-facing value of the refactor. If
the official entry points remain ambiguous, the package is harder to learn and
more fragile to use.

**Independent Test**: Can be fully tested by following the public-facing import
examples and confirming a new user can complete the main PhyloGNN workflow
using only the documented stable package surface.

**Acceptance Scenarios**:

1. **Given** a user starting from the package root, **When** they look for the
   main PhyloGNN workflow imports, **Then** they find a small, clearly curated
   set of official entry points.
2. **Given** a user who only needs the main workflow, **When** they import
   supported top-level objects, **Then** those imports succeed without forcing
   them to understand internal module structure.

---

### User Story 2 - Use Advanced APIs Intentionally (Priority: P2)

As an advanced PhyloGNN user, I want richer package capabilities to remain
available through curated subpackage APIs so I can access training, data, and
model functionality without depending on accidental exports.

**Why this priority**: Advanced users need flexibility, but they also need to
know which subpackage-level imports are intentionally supported.

**Independent Test**: Can be tested independently by verifying that documented
advanced imports remain available from curated subpackage surfaces and that
unsupported low-level names are no longer promoted as package-level APIs.

**Acceptance Scenarios**:

1. **Given** an advanced user working with data, models, or training,
   **When** they import from the documented subpackages, **Then** they receive
   the intended advanced API without relying on internal implementation details.
2. **Given** an advanced user reading package documentation, **When** they
   compare stable, advanced, and internal APIs, **Then** the boundary between
   those layers is explicit.

---

### User Story 3 - Refactor Internals Safely (Priority: P3)

As a package maintainer, I want internal building blocks and optional subsystems
to be separated from the stable public API so I can refactor internals with
lower risk of breaking users unintentionally.

**Why this priority**: This reduces maintenance cost and supports future
package evolution, but it is secondary to improving the end-user import
experience.

**Independent Test**: Can be tested independently by confirming that internal
classes and optional subsystems are no longer presented as stable package-level
contracts while documented user workflows remain intact.

**Acceptance Scenarios**:

1. **Given** a maintainer updating internal model layers or helpers, **When**
   they change internal implementation details, **Then** the documented stable
   public API remains unaffected unless an intentional public contract change is
   made.
2. **Given** a user who does not need optional tree I/O functionality,
   **When** they use the core PhyloGNN workflow, **Then** optional subsystems do
   not block the primary import experience.

### Edge Cases

- What happens when a user relies on names that were previously exposed
  accidentally but are no longer part of the curated public API?
- How does the package communicate the difference between stable top-level
  imports, advanced subpackage imports, and internal module paths?
- What happens when users need tree feature engineering and graph conversion but
  do not need optional tree-loading functionality?
- How does the package handle public metadata and feature-name contracts that
  were previously mutable or inconsistent?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The package MUST define a stable top-level PhyloGNN API that
  covers the main end-user workflow for feature engineering, graph conversion,
  training configuration, training execution, and primary model use.
- **FR-002**: The package MUST define curated advanced APIs for the data,
  models, and training subpackages so users can access richer functionality
  without depending on accidental exports.
- **FR-003**: The package MUST distinguish stable public API, advanced
  subpackage API, and internal implementation API in a way that is explicit to
  users and maintainers.
- **FR-004**: The package MUST stop promoting low-level building blocks as
  package-level public API unless they are intentionally supported for external
  use.
- **FR-005**: The core top-level import experience MUST remain usable for users
  who do not rely on optional tree-loading functionality.
- **FR-006**: Public metadata and feature-related contracts exposed to users
  MUST be stable, read-only where appropriate, and named consistently across
  documentation and behavior.
- **FR-007**: Public package and subpackage exports MUST match real, supported
  symbols and MUST NOT advertise nonexistent or unsupported names.
- **FR-008**: User-facing documentation and examples for importing PhyloGNN
  MUST align with the curated API layers.
- **FR-009**: The feature MUST preserve the current primary user workflows for
  PhyloGNN while reducing ambiguity about which imports are officially
  supported.

### Key Entities *(include if feature involves data)*

- **Stable API Layer**: The small set of top-level package entry points that
  ordinary users can rely on for the main PhyloGNN workflow.
- **Advanced Subpackage API Layer**: The curated, still-supported interfaces
  exposed through domain subpackages such as data, models, and training.
- **Internal API Layer**: Low-level implementation details that remain
  importable only through explicit module paths and are not part of the stable
  package contract.
- **Public Metadata Contract**: User-visible version information, feature names,
  and other exposed metadata whose naming and mutability must be intentional and
  stable.

## Scientific And Data Contracts *(mandatory for package, model, graph, or workflow changes)*

- **Input Contracts**: Users interact with this feature primarily through
  documented package and subpackage import paths, public metadata access, and
  the main PhyloGNN workflow described in package documentation and examples.
- **Output Contracts**: The feature provides a curated set of official import
  surfaces, a consistent set of user-visible public names, and an explicit
  separation between stable, advanced, and internal APIs.
- **Failure Modes**: Unsupported or internal names MUST NOT be presented as part
  of the official public contract. If a workflow depends on optional subsystems,
  the package MUST make that boundary explicit rather than letting unrelated
  optional functionality break the core workflow unexpectedly.
- **Reproducibility Notes**: Public import examples, exported symbol lists, and
  package metadata references MUST remain consistent across package docs,
  examples, and tests so that users can reproduce the documented import
  experience reliably.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A new user can identify and use the main PhyloGNN import surface
  for the primary workflow in under 5 minutes by following the package
  documentation alone.
- **SC-002**: All documented top-level imports for the primary PhyloGNN
  workflow succeed consistently in validation tests.
- **SC-003**: All documented advanced subpackage imports succeed consistently in
  validation tests, and undocumented low-level names are no longer represented
  as official package-level APIs.
- **SC-004**: Public metadata and feature-name contracts referenced in
  documentation and tests use one consistent name for each user-visible concept.
- **SC-005**: Users who do not use optional tree-loading functionality can
  still access the documented core package workflow without import failures
  caused by unrelated optional subsystems.

## Assumptions

- Existing main PhyloGNN workflows for feature engineering, graph conversion,
  model training, and primary model use remain in scope and should remain
  available after the refactor.
- Backward compatibility is desirable for the primary documented workflow, but
  accidentally exposed low-level names may be reduced or removed if they are not
  part of the intended public contract.
- Optional tree-loading functionality remains valuable to some users, but it is
  not required to define the default package surface for all users.
- Documentation and examples are expected to be updated wherever public import
  contracts change.

## Validation Plan

- **Unit/Regression Tests**: Validate exported package and subpackage symbol
  sets, public metadata exposure, read-only public metadata behavior where
  applicable, and consistent user-visible naming for public contracts.
- **Integration/Workflow Tests**: Exercise the main documented PhyloGNN import
  workflow from the package root, advanced subpackage import workflows, and the
  core workflow without optional tree-loading dependencies.
- **Manual Verification**: Review top-level and subpackage import examples in
  user-facing documentation to confirm they match the intended API layers.
