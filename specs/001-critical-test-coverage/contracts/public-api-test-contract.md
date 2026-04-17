# Contract: Public API Validation

## Purpose

Define the user-visible package contracts that the expanded test suite must continue to protect.

## In-Scope Public Surfaces

- Top-level `phylognn` curated exports and version metadata.
- Curated subpackage exports in `phylognn.data`, `phylognn.models`, and `phylognn.training`.
- Optional tree I/O boundary exposed through `phylognn.io`.
- Public classes, factories, and metrics intentionally re-exported from those facades.

## Required Assertions

- Exported names remain canonical and intentionally scoped.
- Hidden low-level helpers remain excluded from curated public surfaces unless explicitly promoted.
- Lazy import resolution continues to expose documented objects.
- Optional I/O functionality remains behind the dedicated boundary module rather than leaking into default surfaces.
- Public scientific/data workflows continue to enforce their documented validation behavior and stable observable outputs.

## Failure Expectations

- Missing or renamed public exports must fail tests immediately.
- Incorrect facade leakage of optional or low-level internals must fail tests immediately.
- Invalid trees, tensors, or configuration values presented to covered public behaviors must raise explicit failures rather than silently coercing input.
