# PhyloGNN Examples

This directory is the planned example suite for the current public PhyloGNN API.
The files are scaffolding for a documentation-first layout and will be filled in
after the reset is complete.

## Planned Example Order

1. `feature_engineering.py`
   Planned entry point for building a small in-memory tree and attaching node features.
2. `tree_to_graph.py`
   Planned entry point for converting a featured tree into a PyTorch Geometric `Data` object.
3. `tree_io.py`
   Planned entry point for reading tree input through the optional `phylognn.io` boundary.
4. `single_task_training.py`
   Planned entry point for a small single-task end-to-end training walkthrough.

## Requirements

- Core examples are intended to use the package runtime dependencies from `pyproject.toml`.
- `tree_io.py` is intended to require the optional DendroPy dependency:

```bash
python -m pip install -e ".[beast]"
```

## Run Commands

When the scripts are implemented, they are expected to be run with:

```bash
python examples/feature_engineering.py
python examples/tree_to_graph.py
python examples/tree_io.py
python examples/single_task_training.py
```

## Notes

- `feature_engineering.py` and `tree_to_graph.py` are planned as self-contained examples.
- `tree_io.py` is planned to read from `examples_data/simulated_trees/`.
- `single_task_training.py` is planned to stay intentionally small and focused on clarity.
