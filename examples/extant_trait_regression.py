"""Train a node-level regressor on the carni70 extant-trait data.

The example keeps labels and masks on one PyTorch Geometric ``Data`` object.
The model predicts log-transformed body size for leaf nodes from structural
features and log-transformed geographic range.
"""

from __future__ import annotations

import csv
import inspect
import math
from pathlib import Path
import warnings
from collections.abc import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from ete3 import Tree
from torch import nn
from torch_geometric.data import Data
from torchmetrics import MeanSquaredError, R2Score

from phylognn import TreeFeatureEngineer, TreeToGraphConverter
from phylognn.models import GATNodeRegressor

HIDDEN_DIM = 64
NUM_LAYERS = 2
EPOCHS = 200
LR = 1e-3
NUM_HEADS = 4
SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1
OUTPUT_DIR = "example_outputs"
PREPROCESS_DIM = 32
HEAD_HIDDEN_DIM = 64
DROPOUT_PROB = 0.2

ROOT = Path(__file__).resolve().parents[1]
TREE_PATH = ROOT / "examples_data" / "carni70" / "carni70_tree.nwk"
CSV_PATH = ROOT / "examples_data" / "carni70" / "carni70_data.csv"

MetricPostprocessFn = Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


# [START load_data]
def load_tree_and_traits(
    tree_path: Path | str,
    csv_path: Path | str,
) -> tuple[Tree, dict[str, dict[str, float]]]:
    """Load the Newick tree and species traits from a CSV file."""
    tree = Tree(str(tree_path), format=1)
    trait_dict: dict[str, dict[str, float]] = {}

    with Path(csv_path).open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            species = row.get("species")
            if not species:
                raise ValueError("Trait CSV rows must contain a non-empty 'species' column.")
            trait_dict[species] = {
                "size": float(row["size"]),
                "range": float(row["range"]),
            }

    return tree, trait_dict


# [END load_data]


# [START build_graph]
def build_graph(tree: Tree, trait_dict: dict[str, dict[str, float]]) -> Data:
    """Build a graph with log-range features and masked log-size labels.

    Leaves without a matching CSV row receive ``NaN`` range and target values.
    Internal nodes receive a ``-1.0`` range sentinel and are excluded from
    ``prediction_mask``.
    """
    nodes = list(tree.traverse("preorder"))
    if not any(node.is_leaf() for node in nodes):
        raise ValueError("Cannot build an extant-trait graph: the tree has zero leaf nodes.")

    engineer = TreeFeatureEngineer()
    engineer.add_features(
        tree,
        origin_time=1.0,
        feature_names=["branch_length", "is_tip"],
        rescale=False,
        inplace=True,
    )

    for node in nodes:
        trait = trait_dict.get(node.name) if node.is_leaf() else None
        if node.is_leaf() and trait is None:
            warnings.warn(
                f"Leaf {node.name!r} has no matching trait row; its target is NaN.",
                UserWarning,
                stacklevel=2,
            )
            node.add_feature("range", float("nan"))
        elif not node.is_leaf():
            node.add_feature("range", -1.0)
        else:
            node.add_feature("range", math.log1p(trait["range"]))

    converter = TreeToGraphConverter(
        feature_names=["branch_length", "is_tip", "range"],
        traversal_strategy="preorder",
        append_is_virtual_feature=False,
    )
    data = converter.convert(tree)

    target_values: list[float] = []
    for node in nodes:
        trait = trait_dict.get(node.name) if node.is_leaf() else None
        size = trait["size"] if trait is not None else float("nan")
        target_values.append(math.log1p(size) if math.isfinite(size) else float("nan"))

    data.y = torch.tensor(target_values, dtype=torch.float32)
    data.prediction_mask = ~torch.isnan(data.y)
    return data


# [END build_graph]


def create_masks(
    prediction_mask: torch.Tensor,
    seed: int = SEED,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split valid prediction nodes into deterministic train/validation/test masks."""
    if prediction_mask.dim() != 1 or prediction_mask.dtype != torch.bool:
        raise ValueError("prediction_mask must be a one-dimensional boolean tensor.")

    torch.manual_seed(seed)
    valid_indices = torch.where(prediction_mask)[0]
    num_valid = int(valid_indices.numel())
    if num_valid == 0:
        raise ValueError("Cannot create train/val/test masks: zero valid prediction nodes.")

    shuffled_indices = valid_indices[torch.randperm(num_valid)]
    num_train = int(TRAIN_RATIO * num_valid)
    num_val = int(VAL_RATIO * num_valid)
    train_indices = shuffled_indices[:num_train]
    val_indices = shuffled_indices[num_train : num_train + num_val]
    test_indices = shuffled_indices[num_train + num_val :]

    train_mask = torch.zeros_like(prediction_mask)
    val_mask = torch.zeros_like(prediction_mask)
    test_mask = torch.zeros_like(prediction_mask)
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    if not torch.equal(train_mask | val_mask | test_mask, prediction_mask):
        raise RuntimeError("Generated train/val/test masks do not cover prediction_mask.")
    return train_mask, val_mask, test_mask


def get_mask(data: Data, mask_key: str | None) -> torch.Tensor:
    """Resolve a named mask, falling back to ``prediction_mask`` when needed."""
    if mask_key is None:
        if hasattr(data, "prediction_mask") and data.prediction_mask is not None:
            return data.prediction_mask
        raise AttributeError(
            "Data object has no 'prediction_mask' attribute. "
            "Ensure prediction_mask is set before training."
        )

    if hasattr(data, mask_key):
        mask = getattr(data, mask_key)
        if mask is not None:
            return mask
    if hasattr(data, "prediction_mask") and data.prediction_mask is not None:
        return data.prediction_mask
    raise AttributeError(
        f"Data object has no '{mask_key}' or 'prediction_mask' attribute. "
        "Ensure masks are set before training."
    )


def _node_predictions(model: nn.Module, data: Data) -> torch.Tensor:
    """Return a scalar prediction vector from the model's ``[N, 1]`` output."""
    predictions = model(data)
    if predictions.dim() == 2 and predictions.size(1) == 1:
        return predictions[:, 0]
    if predictions.dim() == 1:
        return predictions
    raise ValueError(
        "GATNodeRegressor must return a tensor shaped [num_nodes, 1] for this example."
    )


def _r2_score_or_nan(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Return TorchMetrics R2, or NaN when fewer than two samples are available."""
    if targets.numel() < 2:
        return float("nan")
    return float(R2Score()(predictions, targets).item())


def _expm1_metric_postprocess(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Invert the example's log1p transform before calculating metrics."""
    return torch.expm1(predictions), torch.expm1(targets)


def _apply_metric_postprocess(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    metric_postprocess_fn: Callable[..., object] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a metric callback once to complete masked prediction/target tensors.

    The preferred callback contract is ``fn(predictions, targets)`` returning a
    two-tensor tuple.  For convenience, a one-argument transform such as
    ``torch.expm1`` is also accepted and is called once with a stacked
    ``[2, num_values]`` tensor.
    """
    callback = metric_postprocess_fn or _expm1_metric_postprocess
    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        signature = None

    if signature is None:
        accepts_two_arguments = True
    else:
        positional_parameters = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
            and parameter.name != "self"
        ]
        accepts_two_arguments = (
            any(
                parameter.kind is parameter.VAR_POSITIONAL
                for parameter in signature.parameters.values()
            )
            or len(positional_parameters) >= 2
        )

    if accepts_two_arguments:
        processed = callback(predictions, targets)
    else:
        processed = callback(torch.stack((predictions, targets), dim=0))

    if isinstance(processed, tuple) and len(processed) == 2:
        processed_predictions, processed_targets = processed
    elif isinstance(processed, torch.Tensor) and processed.shape == (2, *predictions.shape):
        processed_predictions, processed_targets = processed[0], processed[1]
    else:
        raise TypeError(
            "metric_postprocess_fn must return (predictions, targets) or a stacked "
            "tensor with shape [2, num_values]."
        )

    if not isinstance(processed_predictions, torch.Tensor) or not isinstance(
        processed_targets, torch.Tensor
    ):
        raise TypeError("metric_postprocess_fn must return tensors for predictions and targets.")
    return processed_predictions, processed_targets


# [START train_model]
def train_model(
    model: nn.Module,
    data: Data,
    epochs: int,
    lr: float,
    metric_postprocess_fn: Callable[..., object] | None = None,
) -> dict[str, list[float]]:
    """Train with masked loss and postprocessed validation metrics.

    ``metric_postprocess_fn`` is called once per epoch after collecting all
    masked validation predictions and targets. Loss remains on the original
    model-output scale.
    """
    train_mask = get_mask(data, "train_mask")
    val_mask = get_mask(data, "val_mask")
    if not bool(train_mask.any()):
        raise ValueError("Training mask contains zero valid prediction nodes.")
    if not bool(val_mask.any()):
        raise ValueError("Validation mask contains zero valid prediction nodes.")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "extant_trait_regression_best.pt"

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_mse": [],
        "val_r2": [],
    }
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        predictions = _node_predictions(model, data)
        loss = criterion(predictions[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_predictions = _node_predictions(model, data)
            val_loss = criterion(
                validation_predictions[val_mask],
                data.y[val_mask],
            )
            processed_predictions, processed_targets = _apply_metric_postprocess(
                validation_predictions[val_mask],
                data.y[val_mask],
                metric_postprocess_fn,
            )
            val_mse = MeanSquaredError()(processed_predictions, processed_targets)
            val_r2 = _r2_score_or_nan(processed_predictions, processed_targets)

        train_loss_value = float(loss.item())
        val_loss_value = float(val_loss.item())
        history["train_loss"].append(train_loss_value)
        history["val_loss"].append(val_loss_value)
        history["val_mse"].append(float(val_mse.item()))
        history["val_r2"].append(val_r2)

        if val_loss_value < best_val_loss:
            best_val_loss = val_loss_value
            torch.save(model.state_dict(), checkpoint_path)

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:3d}/{epochs} "
                f"train_loss={train_loss_value:.4f} "
                f"val_loss={val_loss_value:.4f} "
                f"val_mse={history['val_mse'][-1]:.4f} "
                f"val_r2={history['val_r2'][-1]:.4f}"
            )

    if not checkpoint_path.exists():
        torch.save(model.state_dict(), checkpoint_path)
    return history


# [END train_model]


def evaluate_model(
    model: nn.Module,
    data: Data,
    mask_key: str | None,
    metric_postprocess_fn: Callable[..., object] | None = None,
) -> dict[str, float]:
    """Evaluate postprocessed MSE and R2 on the selected mask.

    The callback is applied once to the complete masked prediction and target
    tensors before either metric is computed.
    """
    mask = get_mask(data, mask_key)
    if not bool(mask.any()):
        raise ValueError("Evaluation mask contains zero valid prediction nodes.")

    model.eval()
    with torch.no_grad():
        predictions = _node_predictions(model, data)
        processed_predictions, processed_targets = _apply_metric_postprocess(
            predictions[mask],
            data.y[mask],
            metric_postprocess_fn,
        )
        mse = MeanSquaredError()(processed_predictions, processed_targets)
        r2 = _r2_score_or_nan(processed_predictions, processed_targets)
    return {"mse": float(mse.item()), "r2": r2}


def plot_loss_curves(history: dict[str, list[float]], output_dir: str | Path) -> None:
    """Save train and validation loss curves to the fixed example filename."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots()
    axis.plot(history["train_loss"], label="Train loss")
    axis.plot(history["val_loss"], label="Validation loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("MSE loss (log size)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path / "extant_trait_regression_loss.png")
    plt.close(figure)


def plot_scatter(model: nn.Module, data: Data, output_dir: str | Path) -> None:
    """Save postprocessed test predictions with a dashed identity line."""
    mask = get_mask(data, "test_mask")
    if not bool(mask.any()):
        raise ValueError("Test mask contains zero valid prediction nodes.")

    model.eval()
    with torch.no_grad():
        predictions = torch.expm1(_node_predictions(model, data)[mask]).cpu().numpy()
        targets = torch.expm1(data.y[mask]).cpu().numpy()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    lower = min(float(targets.min()), float(predictions.min()))
    upper = max(float(targets.max()), float(predictions.max()))
    figure, axis = plt.subplots()
    axis.scatter(targets, predictions, alpha=0.7)
    axis.plot([lower, upper], [lower, upper], "k--")
    axis.set_xlabel("Actual size")
    axis.set_ylabel("Predicted size")
    figure.tight_layout()
    figure.savefig(output_path / "extant_trait_regression_scatter.png")
    plt.close(figure)


def main() -> None:
    """Run the complete single-tree node regression workflow."""
    torch.manual_seed(SEED)
    tree, trait_dict = load_tree_and_traits(TREE_PATH, CSV_PATH)
    data = build_graph(tree, trait_dict)
    train_mask, val_mask, test_mask = create_masks(data.prediction_mask, seed=SEED)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    model = GATNodeRegressor(
        input_dim=data.x.size(1),
        output_dim=1,
        preprocess_dim=PREPROCESS_DIM,
        gat_hidden_dim=HIDDEN_DIM,
        gat_heads=NUM_HEADS,
        num_gat_layers=NUM_LAYERS,
        dropout_prob=DROPOUT_PROB,
        head_hidden_dim=HEAD_HIDDEN_DIM,
    )
    history = train_model(model, data, EPOCHS, LR)

    checkpoint_path = Path(OUTPUT_DIR) / "extant_trait_regression_best.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    test_metrics = evaluate_model(model, data, "test_mask")
    plot_loss_curves(history, OUTPUT_DIR)
    plot_scatter(model, data, OUTPUT_DIR)

    print("Extant trait regression summary")
    print(f"valid species: {int(data.prediction_mask.sum().item())}")
    print(f"graph nodes: {data.num_nodes}")
    print(
        f"train/val/test nodes: {int(train_mask.sum())}/{int(val_mask.sum())}/{int(test_mask.sum())}"
    )
    print(f"test MSE: {test_metrics['mse']:.4f}")
    print(f"test R2: {test_metrics['r2']:.4f}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"loss plot: {Path(OUTPUT_DIR) / 'extant_trait_regression_loss.png'}")
    print(f"scatter plot: {Path(OUTPUT_DIR) / 'extant_trait_regression_scatter.png'}")


if __name__ == "__main__":
    main()
