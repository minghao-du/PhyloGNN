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
from torch_geometric.loader import DataLoader
from torchmetrics import MeanSquaredError, R2Score

from phylognn import TreeFeatureEngineer, TreeToGraphConverter, attach_node_targets
from phylognn.models import GATNodeRegressor
from phylognn.training import Trainer, TrainingConfig

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


def _is_integer_dtype(dtype: torch.dtype) -> bool:
    """Return whether ``dtype`` is one of PyTorch's integer tensor dtypes."""
    return dtype in {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}


def validate_graph_data(data: Data) -> None:
    """Validate graph tensors, finite targets, and non-empty disjoint splits."""
    if not isinstance(data, Data):
        raise TypeError(
            f"data must be a torch_geometric.data.Data object, got {type(data).__name__}."
        )
    required_fields = (
        "x",
        "edge_index",
        "y",
        "prediction_mask",
        "train_mask",
        "val_mask",
        "test_mask",
    )
    for field in required_fields:
        if not hasattr(data, field) or getattr(data, field) is None:
            raise ValueError(f"Graph field '{field}' is required before training.")

    x = data.x
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be a Tensor, got {type(x).__name__}.")
    if x.ndim != 2 or not x.is_floating_point() or x.size(0) == 0:
        raise ValueError(
            "x must be a non-empty two-dimensional floating tensor; "
            f"got dtype={x.dtype}, shape={tuple(x.shape)}."
        )

    edge_index = data.edge_index
    if not isinstance(edge_index, torch.Tensor):
        raise TypeError(f"edge_index must be a Tensor, got {type(edge_index).__name__}.")
    if edge_index.ndim != 2 or edge_index.size(0) != 2 or not _is_integer_dtype(edge_index.dtype):
        raise ValueError(
            "edge_index must have integer dtype and shape [2, E]; "
            f"got dtype={edge_index.dtype}, shape={tuple(edge_index.shape)}."
        )

    y = data.y
    if not isinstance(y, torch.Tensor):
        raise TypeError(f"y must be a Tensor, got {type(y).__name__}.")
    if y.ndim != 1 or y.size(0) != x.size(0) or not y.is_floating_point():
        raise ValueError(
            "y must be a one-dimensional floating tensor with one value per node; "
            f"got dtype={y.dtype}, shape={tuple(y.shape)}, num_nodes={x.size(0)}."
        )

    masks: dict[str, torch.Tensor] = {}
    for field in ("prediction_mask", "train_mask", "val_mask", "test_mask"):
        mask = getattr(data, field)
        if not isinstance(mask, torch.Tensor):
            raise TypeError(f"{field} must be a Tensor, got {type(mask).__name__}.")
        if mask.ndim != 1 or mask.dtype != torch.bool or mask.size(0) != x.size(0):
            raise ValueError(
                f"{field} must be a boolean tensor with shape [{x.size(0)}]; "
                f"got dtype={mask.dtype}, shape={tuple(mask.shape)}."
            )
        masks[field] = mask

    finite_targets = torch.isfinite(y)
    if not torch.equal(masks["prediction_mask"], finite_targets):
        raise ValueError(
            "prediction_mask must identify exactly finite y values; "
            f"finite target count={int(finite_targets.sum())}, "
            f"prediction count={int(masks['prediction_mask'].sum())}."
        )

    train_mask = masks["train_mask"]
    val_mask = masks["val_mask"]
    test_mask = masks["test_mask"]
    if (
        bool((train_mask & val_mask).any())
        or bool((train_mask & test_mask).any())
        or bool((val_mask & test_mask).any())
    ):
        raise ValueError("train_mask, val_mask, and test_mask must be pairwise disjoint.")
    if not torch.equal(train_mask | val_mask | test_mask, masks["prediction_mask"]):
        raise ValueError("train_mask, val_mask, and test_mask must cover prediction_mask exactly.")
    for field, mask in (
        ("train_mask", train_mask),
        ("val_mask", val_mask),
        ("test_mask", test_mask),
    ):
        if not bool(mask.any()):
            raise ValueError(f"{field} must contain at least one prediction node.")


def create_masked_graph_view(data: Data, node_mask: torch.Tensor) -> Data:
    """Keep full graph inputs while narrowing ``y`` to selected finite nodes."""
    if not isinstance(node_mask, torch.Tensor):
        raise TypeError(f"node_mask must be a Tensor, got {type(node_mask).__name__}.")
    if node_mask.ndim != 1 or node_mask.dtype != torch.bool or node_mask.size(0) != data.num_nodes:
        raise ValueError(
            "node_mask must be a one-dimensional boolean tensor with one entry per node; "
            f"got dtype={node_mask.dtype}, shape={tuple(node_mask.shape)}, num_nodes={data.num_nodes}."
        )
    if not bool(node_mask.any()):
        raise ValueError("node_mask must select at least one node.")
    if not hasattr(data, "y") or not isinstance(data.y, torch.Tensor) or data.y.ndim != 1:
        raise ValueError("data.y must be a one-dimensional tensor before creating a graph view.")
    selected_targets = data.y[node_mask]
    if not torch.isfinite(selected_targets).all():
        raise ValueError("node_mask selects non-finite target values.")

    view = data.clone()
    view.node_mask = node_mask.clone()
    view.y = selected_targets.clone()
    return view


class MaskedNodeRegressor(nn.Module):
    """Adapt a full-graph node regressor to a selected target subset."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, data: Data) -> torch.Tensor:
        """Return selected predictions with shape ``[K, 1]``."""
        if not hasattr(data, "node_mask"):
            raise ValueError("MaskedNodeRegressor requires data.node_mask.")
        node_mask = data.node_mask
        if (
            node_mask.ndim != 1
            or node_mask.dtype != torch.bool
            or node_mask.size(0) != data.x.size(0)
        ):
            raise ValueError(
                "data.node_mask must be a one-dimensional boolean tensor matching data.x; "
                f"got dtype={node_mask.dtype}, shape={tuple(node_mask.shape)}, x={tuple(data.x.shape)}."
            )
        predictions = self.model(data)
        if not isinstance(predictions, torch.Tensor):
            raise TypeError(f"base model must return a Tensor, got {type(predictions).__name__}.")
        if predictions.ndim == 1:
            predictions = predictions.unsqueeze(-1)
        if (
            predictions.ndim != 2
            or predictions.size(0) != data.x.size(0)
            or predictions.size(1) != 1
        ):
            raise ValueError(
                "base model must return [num_nodes, 1]; "
                f"got shape={tuple(predictions.shape)}, num_nodes={data.x.size(0)}."
            )
        selected = predictions[node_mask]
        if selected.ndim != 2 or selected.size(1) != 1:
            raise ValueError(
                f"selected predictions must have shape [K, 1], got {tuple(selected.shape)}."
            )
        return selected


def create_training_config(
    *,
    epochs: int = EPOCHS,
    batch_size: int = 1,
    learning_rate: float = LR,
    optimizer: str = "adam",
    save_dir: str | Path | None = None,
) -> TrainingConfig:
    """Build and validate the fixed-default configuration used by the example."""
    if save_dir is None:
        save_dir = OUTPUT_DIR
    if not isinstance(save_dir, (str, Path)):
        raise TypeError(f"save_dir must be a path string or Path, got {type(save_dir).__name__}.")
    if not str(save_dir):
        raise ValueError("save_dir must not be empty.")
    save_path = Path(save_dir)
    if save_path.exists() and not save_path.is_dir():
        raise ValueError(f"save_dir must be a directory path, got existing file {save_path}.")
    config = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        optimizer=optimizer,
        save_dir=str(save_dir),
    )
    config.validate()
    return config


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

    used_names = {node.name for node in nodes if node.is_leaf()}
    for index, node in enumerate(nodes):
        if node.is_leaf() or (node.name and node.name not in used_names):
            used_names.add(node.name)
            continue
        generated_name = f"__internal_node_{index}__"
        while generated_name in used_names:
            generated_name = f"_{generated_name}"
        node.name = generated_name
        used_names.add(generated_name)

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

    leaf_names = {node.name for node in nodes if node.is_leaf()}
    size_records = {
        name: math.log1p(trait["size"]) if math.isfinite(trait["size"]) else float("nan")
        for name, trait in trait_dict.items()
        if name in leaf_names
    }
    return attach_node_targets(
        data,
        size_records,
        node_selector=lambda _index, node_name: node_name in leaf_names,
        missing="mask",
    )


# [END build_graph]


def create_masks(
    prediction_mask: torch.Tensor,
    seed: int = SEED,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split valid prediction nodes into deterministic train/validation/test masks."""
    if not isinstance(prediction_mask, torch.Tensor):
        raise TypeError(f"prediction_mask must be a Tensor, got {type(prediction_mask).__name__}.")
    if prediction_mask.dim() != 1 or prediction_mask.dtype != torch.bool:
        raise ValueError(
            "prediction_mask must be a one-dimensional boolean tensor; "
            f"got dtype={prediction_mask.dtype}, shape={tuple(prediction_mask.shape)}."
        )

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

    if not bool(train_mask.any()) or not bool(val_mask.any()) or not bool(test_mask.any()):
        raise ValueError(
            "Generated train/val/test masks must all be non-empty; "
            f"counts={int(train_mask.sum())}/{int(val_mask.sum())}/{int(test_mask.sum())}."
        )
    if not torch.equal(train_mask | val_mask | test_mask, prediction_mask):
        raise ValueError("Generated train/val/test masks do not cover prediction_mask.")
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
    validate_graph_data(data)

    train_view = create_masked_graph_view(data, train_mask)
    val_view = create_masked_graph_view(data, val_mask)
    train_loader = DataLoader([train_view], batch_size=1, shuffle=False)
    val_loader = DataLoader([val_view], batch_size=1, shuffle=False)

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
    config = create_training_config()
    trainer = Trainer(
        model=MaskedNodeRegressor(model),
        config=config,
        loss_fn=nn.MSELoss(),
        metrics={"mse": "mse"},
    )
    history = trainer.fit(train_loader=train_loader, val_loader=val_loader)
    for field in ("train_loss", "val_loss"):
        values = history.get(field)
        if not isinstance(values, list) or not values:
            raise ValueError(f"Training history field '{field}' must be a non-empty list.")
    if len(history["train_loss"]) != len(history["val_loss"]):
        raise ValueError("Training history train_loss and val_loss must have equal lengths.")

    trainer.load_checkpoint("best_model.pt")
    checkpoint_path = Path(OUTPUT_DIR) / "extant_trait_regression_best.pt"
    torch.save(model.state_dict(), checkpoint_path)
    evaluation_data = data.to(trainer.device)
    test_metrics = evaluate_model(model, evaluation_data, "test_mask")
    plot_loss_curves(history, OUTPUT_DIR)
    plot_scatter(model, evaluation_data, OUTPUT_DIR)

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
