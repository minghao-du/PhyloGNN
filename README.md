# PhyloGNN

PhyloGNN converts phylogenetic trees into PyTorch Geometric graph data and
provides model and training utilities for graph neural network workflows on
those data.

The fastest first-use path is:

1. Install the package in the existing project environment.
2. Read the local Sphinx quickstart.
3. Convert a small tree into a `torch_geometric.data.Data` object.

Install the core package from a checkout:

```bash
python -m pip install -e .
```

Install documentation tooling when you want to build the local docs:

```bash
python -m pip install -e ".[docs]"
python -m sphinx -b html -n -W --keep-going docs/source docs/_build/html
```

Open `docs/source/index.rst` for the source documentation, or browse
`docs/_build/html/index.html` after the HTML build. Start with
`docs/source/installation.rst` and `docs/source/quickstart.rst`; the user guide
covers tree input, graph conversion, feature engineering, training, metrics,
tracking, and troubleshooting.

Maintainers can validate the documentation with:

```bash
python -m sphinx -b doctest -W docs/source docs/_build/doctest
```
