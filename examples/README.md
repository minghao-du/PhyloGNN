# PhyloGNN Examples

This directory is a documentation-first example suite for the current public
PhyloGNN API. Start with the small self-contained demos, then move to the
optional file-loading boundary, the lightweight training walkthrough, and the
complete checkpoint-backed pipeline.

## Recommended Order

1. `feature_engineering.py`
   Self-contained introduction to `TreeFeatureEngineer` on an in-memory tree.
2. `tree_to_graph.py`
   Self-contained conversion from a featured tree to a PyTorch Geometric
   `Data` object.
3. `tree_io.py`
   Optional tree-loading example that reads repository sample data from
   `examples_data/simulated_trees/` through `phylognn.io`.
4. `single_task_training.py`
   Compact end-to-end single-task training example using the public workflow
   entry points.
5. `toml_training_config.py`
   TOML-backed training example using `toml_training_config.toml` for model and
   trainer configuration while building tiny local graph data in Python.
6. `complete_pipeline.py`
   Complete tree-to-prediction example that uses the standard TOML checkpoint
   when present and creates a temporary checkpoint when it is absent.

## Requirements

- Install the package and core runtime dependencies before running the
  self-contained examples. The core install includes model and training runtime
  packages such as torch-scatter and tqdm.
- The examples are intended to be run from the repository root after an
  editable install such as `python -m pip install -e ".[dev]"`.
- `feature_engineering.py` and `tree_to_graph.py` are self-contained and do not
  require repository data files.

## Optional Dependencies

- `tree_io.py` depends on the optional DendroPy-backed tree I/O stack exposed by
  `phylognn.io`.
- The optional dependencies are isolated to file-loading workflows; the
  self-contained demos do not require them.
- Install the tree I/O optional dependency set with:

```bash
python -m pip install -e ".[beast]"
```

- Install the aggregate user workflow extra with `python -m pip install -e ".[all]"`
  when you also need tracking and workflow helper dependencies.

## Run Commands

```bash
python examples/feature_engineering.py
python examples/tree_to_graph.py
python examples/tree_io.py
python examples/single_task_training.py
python examples/toml_training_config.py
python examples/complete_pipeline.py
```

## Expected Outputs

- `feature_engineering.py` prints a `Feature engineering summary` plus a compact
  node-by-node feature listing.
- `tree_to_graph.py` prints a `Graph summary` with tensor shapes and selected
  metadata.
- `tree_io.py` prints a `Tree I/O summary` for a tree loaded from
  `examples_data/simulated_trees/`, or a concise optional-dependency guidance
  message when DendroPy is unavailable.
- `single_task_training.py` prints a `Training summary`, dataset sizes, and a
  prediction sample from a tiny single-task workflow.
- `toml_training_config.py` prints a `TOML training run summary`, trains a tiny
  configured model, and writes `example_outputs/toml_training_config/final_model.pt`
  plus `example_outputs/toml_training_config/history.json`.
- `complete_pipeline.py` prints a `Complete pipeline summary`, graph tensor
  shapes, and a prediction value after loading the TOML training checkpoint or
  a temporary checkpoint created internally.

## Notes

- Keep script output concise and high-signal; longer explanations belong here
  rather than inside the scripts.
- `single_task_training.py` writes checkpoints and history under a temporary
  directory that is removed when the script exits.
- `toml_training_config.toml` defines a representative quickstart setup with a
  tiny non-temporal GAT model and short trainer settings.
