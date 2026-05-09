"""Complete tree-to-prediction workflow using the TOML training checkpoint."""

from pathlib import Path
import sys
import tempfile

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


def _predict_with_checkpoint(graph, checkpoint_dir: Path, checkpoint_name: str) -> float:
    trainer = create_trainer_from_config(
        CONFIG_PATH,
        training_overrides={"save_dir": str(checkpoint_dir), "verbose": False},
    )
    trainer.load_checkpoint(checkpoint_name)
    prediction = trainer.predict([graph], batch_size=1)
    return float(prediction.reshape(-1)[0].item())


def _predict_with_temporary_checkpoint(graph) -> tuple[float, Path]:
    with tempfile.TemporaryDirectory(prefix="phylognn_complete_pipeline_") as temp_dir:
        checkpoint_dir = Path(temp_dir)
        trainer = create_trainer_from_config(
            CONFIG_PATH,
            training_overrides={"save_dir": str(checkpoint_dir), "verbose": False},
        )
        trainer.save_checkpoint("final_model.pt")
        value = _predict_with_checkpoint(graph, checkpoint_dir, "final_model.pt")
        return value, checkpoint_dir / "final_model.pt"


def main() -> None:
    torch.manual_seed(7)
    graph = _build_graph()

    if CHECKPOINT_PATH.is_file():
        value = _predict_with_checkpoint(graph, OUTPUT_DIR, "final_model.pt")
        checkpoint_label = CHECKPOINT_PATH.relative_to(ROOT)
    else:
        value, checkpoint_path = _predict_with_temporary_checkpoint(graph)
        checkpoint_label = checkpoint_path

    print("Complete pipeline summary")
    print(f"checkpoint: {checkpoint_label}")
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
