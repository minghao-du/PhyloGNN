# Quickstart: Critical Test Coverage

## Goal

Implement the feature so contributors can verify that every in-scope public module or API surface under `src/phylognn/` has focused automated coverage and an explicit coverage mapping record.

## Prerequisites

1. Create and activate a virtual environment.
2. Install editable package dependencies with dev extras.
3. Install optional extras if you plan to automate optional tree I/O paths.

## Suggested Workflow

1. Review [spec.md](/Users/Minghao/Research/PhyloGNN/specs/001-critical-test-coverage/spec.md), [plan.md](/Users/Minghao/Research/PhyloGNN/specs/001-critical-test-coverage/plan.md), and [data-model.md](/Users/Minghao/Research/PhyloGNN/specs/001-critical-test-coverage/data-model.md).
2. Inventory all in-scope public modules and public API surfaces under `src/phylognn/`.
3. Map each target to an existing pytest file or create a new focused test module when ownership is unclear.
4. Add or extend tests so each target has:
   - At least one success-path assertion.
   - At least one failure-path or edge-case assertion.
   - Regression-oriented checks for stable public exports or scientific/data contracts.
5. Maintain the coverage mapping artifact as tests are added.
6. Document any rare manual exception with reason, impact, and follow-up plan.

## Validation Steps

1. Run the public-facade and release-contract checks:
   `pytest -q tests/test_public_api.py tests/test_io_api.py tests/test_models_api.py tests/test_utils_api.py tests/test_release_contracts.py tests/test_training_api.py`
2. Run the full test suite in an environment with the scientific stack installed:
   `pytest -q tests`
3. If the full scientific stack is not available locally, treat any skipped test modules as environment limits rather than repository-level exceptions and re-run step 2 in a fully provisioned environment before merge.
4. Verify that [coverage-mapping.md](/Users/Minghao/Research/PhyloGNN/specs/001-critical-test-coverage/coverage-mapping.md) names every in-scope target exactly once and points to real pytest locations.
5. Confirm [manual-exceptions.md](/Users/Minghao/Research/PhyloGNN/specs/001-critical-test-coverage/manual-exceptions.md) remains empty or contains only small, justified, reviewable exceptions.

## Expected Deliverables

- Expanded or reorganized pytest modules under `tests/`
- A coverage mapping artifact linking modules or API surfaces to tests
- Any explicit exception records needed for blocked optional-dependency paths
- Passing targeted validation commands for touched areas
- Repeatable validation commands captured directly in this quickstart

## Contracts

- [coverage-mapping-contract.md](/Users/Minghao/Research/PhyloGNN/specs/001-critical-test-coverage/contracts/coverage-mapping-contract.md)
- [public-api-test-contract.md](/Users/Minghao/Research/PhyloGNN/specs/001-critical-test-coverage/contracts/public-api-test-contract.md)
