# PhyloGNN Examples

This directory contains the recommended entry points for learning the current public
PhyloGNN API.

## Example Order

1. `feature_engineering.py`
   Build a small in-memory tree and attach node features with `TreeFeatureEngineer`.
2. `tree_to_graph.py`
   Convert a featured tree into a PyTorch Geometric `Data` object with `TreeToGraphConverter`.
3. `tree_io.py`
   Read a tree file through the optional `phylognn.io` boundary using repository sample data.
4. `single_task_training.py`
   Run a small single-task end-to-end training example with the public training API.

## Requirements

- Core examples require the package runtime dependencies from `pyproject.toml`.
- `tree_io.py` also requires the optional DendroPy dependency:

```bash
python -m pip install -e ".[beast]"
```

## Run Commands

```bash
python examples/feature_engineering.py
python examples/tree_to_graph.py
python examples/tree_io.py
python examples/single_task_training.py
```

## Notes

- `feature_engineering.py` and `tree_to_graph.py` are self-contained.
- `tree_io.py` reads from `examples_data/simulated_trees/`.
- `single_task_training.py` is intentionally small and optimized for clarity rather than model quality.
