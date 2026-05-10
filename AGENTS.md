# AGENTS.md

This file gives repository-specific guidance for coding agents working in
`/Users/Minghao/Research/PhyloGNN`.

## External Rule Files

- Follow `/Users/Minghao/Code/Codex/my-codex-agent/docs/agent-rules/programming.md` for general cross-language programming guidance.
- Repository-specific rules in this `AGENTS.md` take precedence over the general guidance when they conflict.

## Scope

- This is a Python package for converting phylogenetic trees into PyTorch
  Geometric graph data and training GNN models on those graphs.
- Main package code lives under `src/phylognn/`.
- Tests live under `tests/`.
- Example scripts live under `examples/`.
- Treat `src/phylognn/` and `tests/` as the source of truth when examples and
  implementation disagree.

## Environment And Setup

- Python requirement: `>=3.12`.
- Packaging is defined in `pyproject.toml` using setuptools.
- Core runtime dependencies include `torch`, `torch-geometric`, `ete3`, and
  `numpy`.
- Dev dependencies include `pytest`, `black`, and `ruff`.

On this machine, prefer the existing Conda environment:

```bash
conda activate phylognn
```

If the environment is missing package updates, install them into that active
environment:

```bash
python -m pip install -e ".[dev]"
```

If you need optional dataset or workflow extras in the same environment:

```bash
python -m pip install -e ".[all]"
python -m pip install -e ".[beast]"
```

If the `phylognn` Conda environment is unavailable, stop immediately and inform
the user that the required Conda environment does not exist. Do not create a
new virtual environment or install dependencies elsewhere.

## Build, Lint, And Test Commands

Install editable package with dev tools:

```bash
conda activate phylognn
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

## Project-Specific Conventions

- PyTorch Geometric `Data` objects are core inputs and outputs.
- Be explicit about required fields such as `x`, `edge_index`, `batch`, and
  task-specific attributes.
- Check tensor dimensionality and dtype before use.
- Preserve deterministic ordering when feature order or traversal order matters.
- When adding graph-level metadata, keep names descriptive and consistent with
  current fields like `node_names`, `edge_type`, and `original_num_nodes`.
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
- Python >=3.12 + PyTorch, PyTorch Geometric, stdlib `tomllib`, pytest; no new TOML parser dependency (001-toml-training-config)
- Local TOML files as user input; existing trainer checkpoint/history files remain unchanged (001-toml-training-config)
- Python >=3.12 + PyTorch, PyTorch Geometric, stdlib `tomllib`, pytest; new optional extra `wandb` required only when tracking is enabled (001-wandb-training-logging)
- Existing local checkpoint/history files remain unchanged; external wandb run stores configuration, metrics, and status only (001-wandb-training-logging)
- Python >=3.12 + Existing runtime dependencies PyTorch, PyTorch Geometric, ete3, numpy; documentation dependency Sphinx with built-in `autodoc`, `autosummary`, `napoleon`, `viewcode`, and `doctest` extensions (001-sphinx-docs)
- Repository files only: Sphinx sources in `docs/source`, generated HTML in `docs/_build/html`, doctest output in `docs/_build/doctest`, existing non-user notes preserved under `docs/issues` and `docs/myprompt` (001-sphinx-docs)
- Python >=3.12 + Existing runtime dependencies: PyTorch, PyTorch Geometric, ete3, numpy; stdlib `math`; no new dependency required (001-rescale-time-bins)
- N/A; operates on in-memory ETE trees and PyTorch Geometric `Data` objects (001-rescale-time-bins)
- Python >=3.12 + PyTorch, PyTorch Geometric, ETE3, NumPy, `torch-scatter`, `tqdm`; optional DendroPy, Weights & Biases, Sphinx, pytest, Black, Ruff, and audited workflow helper packages such as pandas when retained (001-complete-pyproject-dependencies)
- Local TOML package metadata in `/Users/Minghao/Research/PhyloGNN/pyproject.toml`; pytest audit classifications stored in repository tests; existing checkpoint/history files unchanged (001-complete-pyproject-dependencies)
- Python >=3.12 + PyTorch, PyTorch Geometric, ETE3, NumPy; no new dependency required (001-auto-time-bin)
- In-memory `torch_geometric.data.Data` objects and existing `torch.save` persistence through `TreeToGraphConverter.save_data()` and `convert_and_save()` (001-auto-time-bin)
- Repository documentation files under `/Users/Minghao/Research/PhyloGNN/docs/source`; generated HTML under `/Users/Minghao/Research/PhyloGNN/docs/_build/html`; no runtime data storage changes (001-docs-visual-polish)
- Python >=3.12 for Sphinx autodoc imports; reStructuredText for documentation content; CSS for presentation + Sphinx built-in extensions (`autodoc`, `autosummary`, `napoleon`, `viewcode`, `doctest`); `furo` documentation theme already declared in `pyproject.toml`; existing runtime dependencies imported by autodoc (`torch`, `torch-geometric`, `ete3`, `numpy`) (001-docs-visual-polish)
- Python >=3.12 + PyTorch, PyTorch Geometric, ETE3, NumPy, `torch-scatter`, `tqdm`, stdlib `tomllib`; documentation tooling uses Sphinx built-in `autodoc`, `autosummary`, `napoleon`, `viewcode`, `doctest`, and the Read the Docs theme package `sphinx-rtd-theme` (001-sphinx-docs-fixes)
- Repository files only: Sphinx sources in `/Users/Minghao/Research/PhyloGNN/docs/source`, generated HTML in `/Users/Minghao/Research/PhyloGNN/docs/_build/html`, doctest output in `/Users/Minghao/Research/PhyloGNN/docs/_build/doctest`, runnable examples in `/Users/Minghao/Research/PhyloGNN/examples`, example outputs in `/Users/Minghao/Research/PhyloGNN/example_outputs` (001-sphinx-docs-fixes)
- Python 3.9+ + Sphinx, PyTorch Geometric, PyTorch (002-fix-docs-examples)
- N/A (Temporary directories for outputs) (002-fix-docs-examples)
- reStructuredText, Python 3.x + Sphinx, PhyloGNN package (001-docs-restructure)
- Documentation files (.rst) (001-docs-restructure)
- Python 3.11, Sphinx + sphinx, github-actions (001-host-sphinx-docs-github)
- Python 3.11 + `torchmetrics`, `torch`, `torch_geometric` (002-torchmetrics-migration)
- Python >=3.12 + PyTorch, PyTorch Geometric, torch-scatter, torchmetrics, ete3, numpy, pytest (003-fix-training-stability)
- Local `.pt` graph files and trainer checkpoint/history files; no storage format migration (003-fix-training-stability)

## Recent Changes
