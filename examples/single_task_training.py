"""Lightweight single-task training example for the public PhyloGNN workflow.

The script builds a tiny in-memory dataset, applies feature engineering and
tree-to-graph conversion, creates deterministic train/val/test splits, and
runs a short single-task training loop. It favors clarity and fast runtime over
scientific realism.

Outputs are written to a temporary directory that is removed after the example
finishes.
"""

from pathlib import Path
import tempfile

import torch
import torch.nn as nn
from ete3 import Tree
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool

from phylognn import Trainer, TrainingConfig, TreeFeatureEngineer, TreeToGraphConverter
from phylognn.training import DatasetSplit, SplitPhyloDataset

ROOT = Path(__file__).resolve().parents[1]
FEATURE_NAMES = ["node_time", "time_bin", "branch_length", "is_tip"]


def build_demo_tree(scale: float) -> Tree:
    return Tree(
        f"((A:{1.0 * scale:.2f},B:{1.4 * scale:.2f})C:{0.4 * scale:.2f},"
        f"D:{1.8 * scale:.2f})root:0.0;",
        format=1,
    )


class ToyGraphRegressor(nn.Module):
    """Minimal graph regressor used only for the example training path."""

    def __init__(self, input_dim: int, hidden_dim: int = 16) -> None:
        super().__init__()
        self.node_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.readout = nn.Linear(hidden_dim, 1)

    def forward(self, data: Data) -> torch.Tensor:
        batch = getattr(
            data,
            "batch",
            torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device),
        )
        node_embeddings = self.node_mlp(data.x)
        graph_embeddings = global_mean_pool(node_embeddings, batch)
        return self.readout(graph_embeddings).squeeze(-1)


def make_graph(
    scale: float,
    index: int,
    engineer: TreeFeatureEngineer,
    converter: TreeToGraphConverter,
) -> Data:
    tree = engineer.add_features(
        build_demo_tree(scale),
        origin_time=4.0 + scale,
        feature_names=FEATURE_NAMES,
        rescale=False,
        inplace=True,
    )
    data = converter.convert(tree, graph_attrs={"sample_id": f"sample_{index:02d}"})
    data.y = torch.tensor([scale], dtype=torch.float32)
    return data


def build_dataset() -> SplitPhyloDataset:
    engineer = TreeFeatureEngineer(num_time_bins=6)
    converter = TreeToGraphConverter(
        feature_names=FEATURE_NAMES,
        add_virtual_nodes=False,
        append_is_virtual_feature=False,
        traversal_strategy=engineer.traversal_strategy,
    )
    scales = [0.8, 0.95, 1.1, 1.25, 1.4, 1.55, 1.7, 1.85]
    graphs = [make_graph(scale, index, engineer, converter) for index, scale in enumerate(scales)]
    labels = torch.tensor([[scale] for scale in scales], dtype=torch.float32)
    sample_ids = [graph.sample_id for graph in graphs]
    return SplitPhyloDataset(data_list=graphs, labels=labels, sample_ids=sample_ids)


def main() -> None:
    torch.manual_seed(7)

    dataset = build_dataset()
    split = DatasetSplit.from_ratios(
        sample_ids=dataset.sample_ids,
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        seed=7,
    )
    subsets = dataset.build_subsets(split)

    with tempfile.TemporaryDirectory(prefix="phylognn_single_task_training_") as temp_dir:
        output_dir = Path(temp_dir)
        model = ToyGraphRegressor(input_dim=len(FEATURE_NAMES), hidden_dim=16)
        config = TrainingConfig(
            epochs=3,
            batch_size=4,
            learning_rate=5e-3,
            weight_decay=0.0,
            scheduler=None,
            early_stopping_patience=None,
            save_dir=str(output_dir),
            verbose=False,
        )
        trainer = Trainer(model=model, config=config, metrics={"rmse": "rmse"})

        history = trainer.fit(
            train_dataset=subsets["train"],
            val_dataset=subsets["val"],
        )
        predictions = trainer.predict(subsets["test"])

        first_prediction = float(predictions[0].detach().cpu().item())
        first_target = float(subsets["test"][0].y.detach().cpu().item())

        print("Training summary")
        print(
            "dataset sizes: "
            f"train={len(subsets['train'])}, val={len(subsets['val'])}, "
            f"test={len(subsets['test'])}"
        )
        print(
            "final losses: "
            f"train={history['train_loss'][-1]:.4f}, val={history['val_loss'][-1]:.4f}"
        )
        print(f"output_dir: {output_dir}")
        print(f"prediction sample: pred={first_prediction:.4f}, target={first_target:.4f}")


if __name__ == "__main__":
    main()
