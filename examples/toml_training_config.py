"""TOML-backed training configuration example with a tiny local dataset."""

from pathlib import Path
import shutil

import torch
from ete3 import Tree

from phylognn import TreeFeatureEngineer, TreeToGraphConverter
from phylognn.training import (
    DatasetSplit,
    SplitPhyloDataset,
    create_trainer_from_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "examples" / "toml_training_config.toml"
OUTPUT_DIR = ROOT / "example_outputs" / "toml_training_config"
FEATURE_NAMES = ("node_time", "time_bin", "branch_length", "is_tip")


def _build_tree(scale: float) -> Tree:
    return Tree(
        f"((A:{0.8 * scale:.2f},B:{1.1 * scale:.2f})C:{0.4 * scale:.2f},"
        f"D:{1.3 * scale:.2f})root:0.0;",
        format=1,
    )


def _make_graph(
    scale: float,
    index: int,
    engineer: TreeFeatureEngineer,
    converter: TreeToGraphConverter,
):
    tree = engineer.add_features(
        _build_tree(scale),
        origin_time=3.5 + scale,
        feature_names=FEATURE_NAMES,
        rescale=False,
        inplace=True,
    )
    return converter.convert(tree, graph_attrs={"sample_id": f"toml_sample_{index:02d}"})


def _build_dataset() -> SplitPhyloDataset:
    engineer = TreeFeatureEngineer(num_time_bins=6)
    converter = TreeToGraphConverter(
        feature_names=FEATURE_NAMES,
        add_virtual_nodes=False,
        append_is_virtual_feature=False,
        traversal_strategy=engineer.traversal_strategy,
    )
    scales = [0.80, 0.95, 1.10, 1.25, 1.40, 1.55, 1.70, 1.85]
    graphs = [_make_graph(scale, index, engineer, converter) for index, scale in enumerate(scales)]
    labels = torch.tensor([[[scale]] for scale in scales], dtype=torch.float32)
    sample_ids = [graph.sample_id for graph in graphs]
    return SplitPhyloDataset(data_list=graphs, labels=labels, sample_ids=sample_ids)


def main() -> None:
    torch.manual_seed(7)
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

    dataset = _build_dataset()
    split = DatasetSplit.from_ratios(
        sample_ids=dataset.sample_ids,
        train_ratio=0.75,
        val_ratio=0.25,
        test_ratio=0.0,
        seed=7,
    )
    subsets = dataset.build_subsets(split)
    trainer = create_trainer_from_config(
        CONFIG_PATH,
        training_overrides={"save_dir": str(OUTPUT_DIR), "verbose": False},
    )
    history = trainer.fit(
        train_dataset=subsets["train"],
        val_dataset=subsets["val"],
    )

    print("TOML training run summary")
    print(f"configured model: {trainer.model.__class__.__name__}")
    print(f"feature order: {FEATURE_NAMES}")
    print(f"dataset sizes: train={len(subsets['train'])}, val={len(subsets['val'])}")
    print(f"epochs: {trainer.config.epochs}")
    print(f"batch_size: {trainer.config.batch_size}")
    print(
        "final losses: " f"train={history['train_loss'][-1]:.4f}, val={history['val_loss'][-1]:.4f}"
    )
    print(f"metrics: {', '.join(trainer.metrics)}")
    print(f"tracking: {'enabled' if trainer.tracking_config.enabled else 'disabled'}")
    print(f"checkpoint: {(OUTPUT_DIR / 'final_model.pt').relative_to(ROOT)}")
    print(f"history: {(OUTPUT_DIR / 'history.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
