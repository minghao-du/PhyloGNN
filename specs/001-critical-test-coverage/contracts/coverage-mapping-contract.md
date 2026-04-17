# Contract: Coverage Mapping Artifact

## Purpose

Define the required structure for the human-readable artifact that proves every in-scope public module or API surface has automated test ownership.

## Required Fields Per Entry

- `Target`: Canonical module path or public API surface name.
- `Tier`: Coverage tier assigned to the target.
- `Public Contract`: Observable contract protected by tests.
- `Test Locations`: Pytest files or sections responsible for the target.
- `Scenario Coverage`: Summary of success-path, failure-path, and regression evidence.
- `Exception`: Empty when fully automated; otherwise a short pointer to the documented exception record.

## Behavioral Rules

- Every in-scope target under `src/phylognn/` must appear exactly once.
- Entries must use canonical package names so contributors can search them directly.
- Test locations must point to real pytest modules or clearly named sections.
- Shared tests are allowed, but the mapping must still make ownership explicit.
- Exception references are allowed only for a small number of blocked cases and must include reason, impact, and follow-up elsewhere in the artifact set.

## Acceptance Signals

- A maintainer can inspect the artifact and determine which tests protect each target without reading source line by line.
- Missing targets, empty scenario coverage, or unresolved exception references are treated as incomplete implementation.
