"""Minimal deterministic training smoke test for the Quickstart guide."""

from pathlib import Path
import tempfile

import torch
import torch.nn as nn
from ete3 import Tree
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool

from phylognn import Trainer, TrainingConfig, TreeFeatureEngineer, TreeToGraphConverter

FEATURE_NAMES = ["node_time", "time_bin", "branch_length", "is_tip"]


# [START build_tree]
def build_tree() -> Tree:
    return Tree("((A:1.0,B:1.5)C:0.5,D:2.0)root:0.0;", format=1)


# [END build_tree]


# [START make_graph]
def make_graph() -> Data:
    engineer = TreeFeatureEngineer(num_time_bins=6)
    tree = engineer.add_features(
        build_tree(),
        origin_time=4.0,
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
    data = converter.convert(tree, graph_attrs={"sample_id": "quickstart"})
    data.y = torch.tensor([1.0], dtype=torch.float32)
    return data


# [END make_graph]


class TinyGraphRegressor(nn.Module):
    """Small graph-level regressor used only by the quickstart smoke test."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.node_mlp = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
        )
        self.readout = nn.Linear(8, 1)

    def forward(self, data: Data) -> torch.Tensor:
        batch = getattr(
            data,
            "batch",
            torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device),
        )
        graph_embedding = global_mean_pool(self.node_mlp(data.x), batch)
        return self.readout(graph_embedding).squeeze(-1)


# [START validate_graph]
def validate_graph(data: Data) -> None:
    assert data.x.dim() == 2
    assert data.x.dtype == torch.float32
    assert data.edge_index.shape[0] == 2
    assert data.edge_index.dtype == torch.long
    assert data.y.shape == (1,)
    assert data.y.dtype == torch.float32


# [END validate_graph]


# [START train_and_predict]
def train_and_predict(data: Data) -> float:
    with tempfile.TemporaryDirectory(prefix="phylognn_quickstart_") as temp_dir:
        model = TinyGraphRegressor(input_dim=data.x.size(1))
        config = TrainingConfig(
            epochs=2,
            batch_size=1,
            learning_rate=1e-2,
            weight_decay=0.0,
            scheduler=None,
            early_stopping_patience=None,
            save_dir=str(Path(temp_dir)),
            save_best_only=False,
            verbose=False,
        )
        trainer = Trainer(model=model, config=config)
        trainer.fit(train_dataset=[data])
        prediction = trainer.predict(dataset=[data])
    return float(prediction[0].detach().cpu().item())


# [END train_and_predict]


def main() -> None:
    torch.manual_seed(11)
    data = make_graph()
    validate_graph(data)
    prediction = train_and_predict(data)

    print("Quickstart training summary")
    print(f"x shape: {tuple(data.x.shape)}")
    print(f"edge_index shape: {tuple(data.edge_index.shape)}")
    print(f"target shape: {tuple(data.y.shape)}")
    print("batch ready: true")
    print(f"prediction: {prediction:.4f}")


if __name__ == "__main__":
    main()
