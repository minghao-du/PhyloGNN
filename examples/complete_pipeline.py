"""Complete tree-to-prediction workflow using the TOML training checkpoint."""

from pathlib import Path
import sys

import torch
from ete3 import Tree

from phylognn import TreeFeatureEngineer, TreeToGraphConverter
from phylognn.training import create_trainer_from_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "examples" / "toml_training_config.toml"
OUTPUT_DIR = ROOT / "example_outputs" / "toml_training_config"
CHECKPOINT_PATH = OUTPUT_DIR / "final_model.pt"
FEATURE_NAMES = ("node_time", "time_bin", "branch_length", "is_tip")


def _build_tree() -> Tree:
    return Tree("((A:0.92,B:1.18)C:0.42,D:1.36)root:0.0;", format=1)


def _build_graph():
    engineer = TreeFeatureEngineer(num_time_bins=6)
    tree = engineer.add_features(
        _build_tree(),
        origin_time=4.2,
        feature_names=FEATURE_NAMES,
        rescale=False,
        inplace=True,
    )
    converter = TreeToGraphConverter(
        feature_names=FEATURE_NAMES,
        add_virtual_nodes=False,
        append_is_virtual_feature=False,
        traversal_strategy=engineer.traversal_strategy,
    )
    return converter.convert(tree, graph_attrs={"sample_id": "pipeline_tree"})


def main() -> None:
    if not CHECKPOINT_PATH.is_file():
        raise SystemExit("Missing checkpoint. Run `python examples/toml_training_config.py` first.")

    torch.manual_seed(7)
    graph = _build_graph()
    trainer = create_trainer_from_config(
        CONFIG_PATH,
        training_overrides={"save_dir": str(OUTPUT_DIR), "verbose": False},
    )
    trainer.load_checkpoint("final_model.pt")
    prediction = trainer.predict([graph], batch_size=1)
    value = float(prediction.reshape(-1)[0].item())

    print("Complete pipeline summary")
    print(f"checkpoint: {CHECKPOINT_PATH.relative_to(ROOT)}")
    print(f"graph x shape: {tuple(graph.x.shape)}")
    print(f"graph edge_index shape: {tuple(graph.edge_index.shape)}")
    print(f"prediction: {value:.4f}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"Complete pipeline failed: {exc}", file=sys.stderr)
        raise
