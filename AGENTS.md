# AGENTS.md

This file gives repository-specific guidance for coding agents working in
`/Users/Minghao/Research/PhyloGNN`.

## Scope

- This is a Python package for converting phylogenetic trees into PyTorch
  Geometric graph data and training GNN models on those graphs.
- Main package code lives under `src/phylognn/`.
- Tests live under `tests/`.
- Example scripts live under `examples/`.
- Treat `src/phylognn/` and `tests/` as the source of truth when examples and
  implementation disagree.

## Repository Rules Files

- No repository Cursor rules were found in `.cursor/rules/`.
- No `.cursorrules` file was found.
- No Copilot instructions file was found at
  `.github/copilot-instructions.md`.
- Therefore, follow this `AGENTS.md` plus the existing codebase conventions.

## Environment And Setup

- Python requirement: `>=3.8`.
- Packaging is defined in `pyproject.toml` using setuptools.
- Core runtime dependencies include `torch`, `torch-geometric`, `ete3`, and
  `numpy`.
- Dev dependencies include `pytest`, `black`, and `ruff`.

Recommended setup:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If you need optional dataset or workflow extras:

```bash
python -m pip install -e ".[all]"
python -m pip install -e ".[beast]"
```

## Build, Lint, And Test Commands

Install editable package with dev tools:

```bash
python -m pip install -e ".[dev]"
```

Run the full test suite:

```bash
pytest
```

Run a single test file:

```bash
pytest tests/test_feature_engineer.py
```

Run a single test class:

```bash
pytest tests/test_feature_engineer.py::TestAddFeatures
```

Run a single test function:

```bash
pytest tests/test_feature_engineer.py::TestAddFeatures::test_add_features_inplace
```

Run tests by name pattern:

```bash
pytest -k rescale_tree
```

Run tests with verbose output or stop on first failure:

```bash
pytest -v
pytest -x
```

Run lint checks:

```bash
ruff check src tests examples
black --check src tests examples
```

Auto-format code:

```bash
black src tests examples
```

Build a distributable package:

```bash
python -m build
```

If `python -m build` is unavailable, install it first:

```bash
python -m pip install build
python -m build
```

## Project Layout

- `src/phylognn/data/`: tree feature engineering, conversion, tree I/O.
- `src/phylognn/models/`: GNN layers and model definitions.
- `src/phylognn/training/`: dataset abstractions, metrics, trainer utilities.
- `src/phylognn/utils/`: small helper utilities.
- `tests/`: pytest-based unit tests.

## General Coding Style

- Follow Black formatting with line length 100.
- Follow Ruff linting with line length 100.
- Use 4-space indentation.
- Prefer ASCII unless a file already uses another language or Unicode symbols.
- Keep public APIs explicit and stable.
- Prefer readability and explicit contracts over cleverness.

## Imports

- Group imports in this order:
  1. future imports
  2. standard library
  3. third-party packages
  4. local package imports
- Separate groups with a single blank line.
- Prefer explicit imports over wildcard imports.
- Use relative imports within a package module when importing siblings.
- Keep import lists readable; use parenthesized multi-line imports when needed.

## Typing

- Add type hints to public functions, methods, and class attributes.
- This codebase uses `Optional`, `Union`, `Literal`, `Sequence`, `Mapping`, and
  `Tuple` heavily; stay consistent with existing style.
- For reusable signatures, create type aliases near the top of the module.
- Use `from __future__ import annotations` in modules that benefit from forward
  references or modern annotation behavior.
- Return concrete, predictable types.
- Validate runtime assumptions even when types are present.

## Naming Conventions

- Classes: `PascalCase`.
- Functions and methods: `snake_case`.
- Variables and attributes: `snake_case`.
- Constants: `UPPER_SNAKE_CASE`.
- Type aliases use descriptive `PascalCase` names such as `PathLike` or
  `ModelOutput`.
- Test classes start with `Test...` and test methods start with `test_...`.

## Docstrings And Documentation

- The repository strongly favors substantial docstrings, especially in core
  modules under `src/phylognn/`.
- Use triple-double-quoted docstrings.
- For public classes and methods, document parameters, behavior, return values,
  and raised exceptions when the behavior is non-trivial.
- Explain data contracts clearly, especially tensor shapes, graph fields, and
  feature semantics.
- Keep comments sparse; prefer clear code and docstrings.

## Error Handling And Validation

- Validate inputs early.
- Raise `ValueError` for invalid values or inconsistent state.
- Raise `TypeError` for wrong object types or wrong tensor dtypes.
- Error messages should be explicit and actionable.
- Existing code frequently validates dimensions, allowed literal values,
  positivity constraints, and required attributes; preserve that pattern.
- Do not silently coerce invalid inputs unless there is an established reason.

## Data And Tensor Conventions

- PyTorch Geometric `Data` objects are core inputs and outputs.
- Be explicit about required fields such as `x`, `edge_index`, `batch`, and
  task-specific attributes.
- Check tensor dimensionality and dtype before use.
- Preserve deterministic ordering when feature order or traversal order matters.
- When adding graph-level metadata, keep names descriptive and consistent with
  current fields like `node_names`, `edge_type`, and `original_num_nodes`.

## API Design Preferences

- Prefer small, composable helpers plus a clearly documented public method.
- Keep feature engineering, graph conversion, model logic, and training logic
  separated by responsibility.
- Reuse existing abstractions before adding new parallel ones.
- Preserve backward-compatible parameter names unless the task explicitly calls
  for breaking changes.
- When adding options, validate them centrally with dedicated helper methods.

## Testing Expectations

- Add or update pytest coverage for any non-trivial behavioral change.
- Prefer focused unit tests over broad integration tests unless needed.
- Match the existing style of direct assertions and `pytest.raises(...)`.
- Use parametrization for repeated validation cases.
- When fixing a bug, add a regression test close to the affected module area.

## Working In This Repository

- Check whether a similar helper, validator, or type alias already exists before
  introducing a new one.
- Favor consistency with `src/phylognn/data/`, `src/phylognn/models/`, and
  `src/phylognn/training/`; these modules show the clearest current standards.
- Be cautious with `examples/`: some example code appears older and may not
  reflect the latest API exactly.
- If updating exports in `__init__.py`, keep `__all__` accurate and consistent.
- Avoid unrelated refactors unless they are necessary for correctness.

## Agent Checklist

- Read the target module and nearby tests before editing.
- Prefer minimal, local changes that fit existing abstractions.
- Run relevant tests after code changes; for narrow changes, run a single test,
  class, or file first.
- Run `ruff check` and `black --check` on touched areas when practical.
- If you add a new public API or behavior contract, update docstrings and tests.

## Active Technologies
- Python >=3.8 + PyTorch, PyTorch Geometric, ETE3, NumPy; optional tree I/O path may use DendroPy via extras (001-api-exposure-refactor)
- Source files, package metadata, and optional serialized `.pt` graph artifacts; no database (001-api-exposure-refactor)
- Python >=3.8 + PyTorch, PyTorch Geometric, ETE3, NumPy; optional DendroPy for BEAST/tree I/O paths (001-critical-test-coverage)
- Source files, pytest fixtures, and Markdown planning artifacts in the repository; no database (001-critical-test-coverage)

## Recent Changes
- 001-api-exposure-refactor: Added Python >=3.8 + PyTorch, PyTorch Geometric, ETE3, NumPy; optional tree I/O path may use DendroPy via extras
