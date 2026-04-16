# Research: API Exposure Refactor

## Decision 1: Keep the existing package framework and dependency stack

- Decision: Implement the refactor inside the current `src/phylognn/` package
  structure and continue using the repository's existing Python, PyTorch, and
  PyTorch Geometric stack.
- Rationale: The user explicitly requested keeping the current framework, and
  the feature concerns API curation rather than architectural replacement.
  Staying within the current package structure minimizes risk and aligns with
  the constitution's pragmatic-simplicity principle.
- Alternatives considered:
  - Introduce a new facade package layer: rejected because it adds extra
    indirection without solving the current ambiguity cleanly.
  - Reorganize the repository into multiple packages: rejected because the
    refactor scope is export hygiene, not packaging redesign.

## Decision 2: Model the user-facing contract as three API layers

- Decision: Use a three-layer contract:
  - stable top-level API in `phylognn`
  - curated advanced subpackage APIs in `phylognn.data`,
    `phylognn.models`, and `phylognn.training`
  - internal module-path-only APIs for low-level building blocks
- Rationale: This matches the feature spec, improves user discoverability, and
  gives maintainers a clear boundary for future refactors.
- Alternatives considered:
  - Keep the current partially implicit surface: rejected because it preserves
    accidental exports and weakens contract stability.
  - Expose every reusable symbol at package level: rejected because it makes
    future model and training refactors unnecessarily risky.

## Decision 3: Keep optional tree I/O out of the default root import path

- Decision: Decouple tree-loading helpers from root-package imports so users can
  access the main preprocessing and training workflow without pulling optional
  tree I/O dependencies into the default import path. Expose that functionality
  from the dedicated `phylognn.io` module instead.
- Rationale: The current plan identifies root-import coupling to tree I/O as a
  primary problem. This change directly improves package robustness for users
  who only need feature engineering, graph conversion, training, or model use.
- Alternatives considered:
  - Continue importing tree I/O from `phylognn.data` and `phylognn` by default:
    rejected because it keeps optional dependencies coupled to the main API.
  - Remove tree I/O entirely from the public product surface: rejected because
    some users still need intentional access to tree loading.

## Decision 4: Curate exports instead of building a compatibility layer first

- Decision: Fix `__all__` definitions, remove nonexistent exports, add missing
  intended exports, standardize `__version__`, and restrict low-level model
  layers from package-level exports before introducing any larger compatibility
  framework.
- Rationale: These are the highest-value, lowest-risk steps and match the safe
  cleanup phase in the refactor proposal.
- Alternatives considered:
  - Add a deprecation shim for all accidental exports up front: rejected because
    it increases implementation surface before validating actual breakage risk.
  - Defer export cleanup until after a larger redesign: rejected because the
    current ambiguity is already a user-facing problem.

## Decision 5: Validate the public API through import-surface tests

- Decision: Add focused tests for top-level and subpackage imports, metadata
  exposure, optional-dependency decoupling, and consistent public naming.
- Rationale: This is the smallest practical evidence set that proves the new
  public contract without overengineering the test suite.
- Alternatives considered:
  - Rely only on manual import checks: rejected because the feature changes
    explicit package contracts and needs regression protection.
  - Build large end-to-end training tests first: rejected because they are
    slower and not the most direct evidence for API-surface correctness.
