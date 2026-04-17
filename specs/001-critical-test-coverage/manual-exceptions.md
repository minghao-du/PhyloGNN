# Manual Exceptions: Critical Test Coverage

## Policy

Manual exceptions are allowed only when a behavior cannot be automated without unsupported optional dependencies or unstable external setup. Every exception must record:

1. The blocked target
2. The specific blocked behavior
3. The reason automation is currently blocked
4. The impact on release confidence
5. The exact manual verification step
6. The follow-up plan for removing the exception

## Template

| Target | Blocked Behavior | Reason | Impact | Manual Verification | Follow-Up Plan |
|--------|------------------|--------|--------|---------------------|----------------|
| _example_ | _Specific uncovered contract_ | _Why automation is blocked_ | _What remains at risk_ | _How to verify manually_ | _How to remove exception later_ |

## Current Status

No feature-level exceptions are accepted by default. Add entries here only if implementation confirms a small, justified gap remains after automated test work is complete.

## Validation Notes

- Local validation on 2026-04-17 passed the dependency-light facade and release contract tests and passed the repository test suite with scientific-stack-dependent modules skipped because `torch`, `torch_geometric`, `ete3`, `dendropy`, and `torch_scatter` were not installed in the active environment.
- Re-run `pytest -q tests` in a fully provisioned scientific Python environment before merge to convert those skips into full execution coverage.
