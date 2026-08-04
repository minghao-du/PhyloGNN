"""In-memory evaluation of one leaf-aligned region on one phylogenetic tree."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Real

import torch
from ete3 import Tree

from .models.masked_attention import MaskedAttentionPhyloRegressor


@dataclass(frozen=True)
class RegionAssociationResult:
    """Scores and attention returned by :func:`evaluate_region_association`.

    Tensor fields are detached clones of evaluator state. The dataclass prevents
    field reassignment, but does not make the returned tensors deeply immutable.
    """

    cv_r2: float
    fold_r2: tuple[float, ...]
    attention: torch.Tensor
    mean_attention: torch.Tensor


def _resolve_leaf_names(tree: Tree, leaf_names: Sequence[str] | None) -> tuple[str, ...]:
    if not isinstance(tree, Tree):
        raise TypeError("`tree` must be an ete3.Tree.")
    leaves = list(tree.iter_leaves())
    tree_names = tuple(leaf.name for leaf in leaves)
    if not tree_names or any(not isinstance(name, str) or not name.strip() for name in tree_names):
        raise ValueError("Tree leaves must have nonblank names.")
    if len(set(tree_names)) != len(tree_names):
        raise ValueError("Tree leaf names must be unique.")
    if leaf_names is None:
        return tree_names

    if isinstance(leaf_names, (str, bytes)) or not isinstance(leaf_names, Sequence):
        raise TypeError("`leaf_names` must be a sequence of strings.")
    names = tuple(leaf_names)
    if any(not isinstance(name, str) for name in names):
        raise TypeError("`leaf_names` must be a sequence of strings.")
    if (
        len(names) != len(tree_names)
        or len(set(names)) != len(names)
        or set(names) != set(tree_names)
        or any(not name.strip() for name in names)
    ):
        raise ValueError("`leaf_names` must be a complete unique permutation of tree leaf names.")
    return names


def build_leaf_laplacian(tree: Tree, leaf_names: Sequence[str] | None = None) -> torch.Tensor:
    """Build a deterministic float32 normalized Laplacian over tree leaves.

    Args:
        tree: ETE tree with unique, nonblank leaf names.
        leaf_names: Optional complete permutation defining output row order.

    Returns:
        A finite float32 tensor of shape ``[N, N]``.
    """
    names = _resolve_leaf_names(tree, leaf_names)
    leaves_by_name = {leaf.name: leaf for leaf in tree.iter_leaves()}
    count = len(names)
    distances = torch.zeros((count, count), dtype=torch.float32)
    for row in range(count):
        for column in range(row + 1, count):
            distance = float(leaves_by_name[names[row]].get_distance(leaves_by_name[names[column]]))
            if not math.isfinite(distance) or distance < 0:
                raise ValueError("Leaf path distances must be finite and non-negative.")
            distances[row, column] = distance
            distances[column, row] = distance

    if not torch.isfinite(distances).all():
        raise ValueError("Leaf path distances must produce finite float32 values.")

    if count > 1 and torch.count_nonzero(distances) == 0:
        for row in range(count):
            for column in range(row + 1, count):
                distance = float(
                    leaves_by_name[names[row]].get_distance(
                        leaves_by_name[names[column]], topology_only=True
                    )
                )
                if not math.isfinite(distance) or distance < 0:
                    raise ValueError("Leaf topology distances must be finite and non-negative.")
                distances[row, column] = distance
                distances[column, row] = distance

    if not torch.isfinite(distances).all():
        raise ValueError("Leaf topology distances must produce finite float32 values.")

    positive_distances = distances[distances > 0]
    scale = positive_distances.mean() if positive_distances.numel() else torch.tensor(1.0)
    similarity = torch.exp(-distances / scale)
    similarity.fill_diagonal_(0.0)
    degrees = similarity.sum(dim=1)
    inverse_sqrt_degrees = torch.zeros_like(degrees)
    nonzero_degree = degrees > 0
    inverse_sqrt_degrees[nonzero_degree] = degrees[nonzero_degree].rsqrt()
    laplacian = torch.eye(count, dtype=torch.float32) - (
        inverse_sqrt_degrees[:, None] * similarity * inverse_sqrt_degrees[None, :]
    )
    if not torch.isfinite(laplacian).all():
        raise ValueError("`leaf_laplacian` construction produced non-finite values.")
    return laplacian


def _as_floating_tensor(value: object, field_name: str) -> torch.Tensor:
    """Convert tensor-like floating values while preserving invalid-input failures."""
    try:
        tensor = torch.as_tensor(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"`{field_name}` must be tensor-like floating data.") from error
    if not tensor.is_floating_point():
        raise ValueError(f"`{field_name}` must contain floating-point values; got {tensor.dtype}.")
    return tensor.to(dtype=torch.float32)


def _as_position_mask(value: object) -> torch.Tensor:
    """Convert a tensor-like mask to boolean values with a clear contract error."""
    try:
        return torch.as_tensor(value, dtype=torch.bool)
    except (TypeError, ValueError) as error:
        raise TypeError("`position_mask` must be tensor-like and bool-convertible.") from error


def _validate_positive_int(value: object, field_name: str) -> int:
    """Return a positive integer setting, excluding booleans."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"`{field_name}` must be a positive integer.")
    return value


def _validate_nonnegative_finite_real(value: object, field_name: str) -> float:
    """Return a finite real setting at or above zero, excluding booleans."""
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"`{field_name}` must be a finite non-negative real number.")
    return float(value)


def _as_targets(targets: Mapping[str, float] | object, leaf_names: tuple[str, ...]) -> torch.Tensor:
    if isinstance(targets, Mapping):
        if set(targets) != set(leaf_names):
            raise ValueError("Target mapping keys must exactly match `leaf_names`.")
        values = [targets[name] for name in leaf_names]
        result = _as_floating_tensor(values, "targets")
    else:
        result = _as_floating_tensor(targets, "targets")
    if result.ndim != 1 or result.numel() != len(leaf_names):
        raise ValueError(
            "`targets` must have shape [N] matching the tree leaf count; "
            f"got {tuple(result.shape)}."
        )
    if not torch.isfinite(result).all():
        raise ValueError("`targets` must contain only finite values.")
    return result


def _fit_model(
    model: MaskedAttentionPhyloRegressor,
    representations: torch.Tensor,
    position_mask: torch.Tensor,
    targets: torch.Tensor,
    train_indices: torch.Tensor,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for _ in range(epochs):
        optimizer.zero_grad()
        predictions, _ = model(representations, position_mask)
        loss = torch.nn.functional.mse_loss(predictions[train_indices], targets[train_indices])
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return model(representations, position_mask)


def evaluate_region_association(
    tree: Tree,
    representations: torch.Tensor,
    position_mask: torch.Tensor,
    targets: Mapping[str, float] | object,
    *,
    leaf_names: Sequence[str] | None = None,
    n_splits: int = 5,
    epochs: int = 100,
    hidden_dim: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    seed: int = 0,
) -> RegionAssociationResult:
    """Cross-validate and refit a masked-attention region regressor in memory.

    Every fold makes a full-tree forward pass but computes loss and R-squared
    only on its train and validation leaves respectively. The final refit uses
    all targets and provides the returned attention.
    """
    names = _resolve_leaf_names(tree, leaf_names)
    representations = _as_floating_tensor(representations, "representations")
    position_mask = _as_position_mask(position_mask)
    if (
        representations.ndim != 3
        or representations.size(0) != len(names)
        or representations.size(1) == 0
        or representations.size(2) == 0
    ):
        raise ValueError(
            "`representations` must have nonempty shape [N, L, D] aligned to tree leaves; "
            f"got {tuple(representations.shape)}."
        )
    if not torch.isfinite(representations).all():
        raise ValueError("`representations` must contain only finite values.")
    if position_mask.shape != representations.shape[:2]:
        raise ValueError(
            "`position_mask` must have shape [N, L] matching `representations`; "
            f"got {tuple(position_mask.shape)}."
        )
    if not torch.all(position_mask.any(dim=1)):
        raise ValueError("Every `position_mask` row must contain a valid position.")
    targets_tensor = _as_targets(targets, names)
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("`n_splits` must be an integer of at least two.")
    if len(names) // n_splits < 2:
        raise ValueError("`n_splits` must leave at least two validation leaves per fold.")
    epochs = _validate_positive_int(epochs, "epochs")
    hidden_dim = _validate_positive_int(hidden_dim, "hidden_dim")
    learning_rate = _validate_nonnegative_finite_real(learning_rate, "learning_rate")
    if learning_rate == 0:
        raise ValueError("`learning_rate` must be a positive finite real number.")
    weight_decay = _validate_nonnegative_finite_real(weight_decay, "weight_decay")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("`seed` must be an integer.")

    laplacian = build_leaf_laplacian(tree, names)
    generator = torch.Generator().manual_seed(seed)
    folds = torch.tensor_split(torch.randperm(len(names), generator=generator), n_splits)
    for validation_indices in folds:
        validation_targets = targets_tensor[validation_indices]
        denominator = torch.sum((validation_targets - validation_targets.mean()) ** 2)
        if denominator == 0:
            raise ValueError("Validation fold has constant targets, so R-squared is undefined.")
    fold_scores: list[float] = []

    with torch.random.fork_rng(devices=[]):
        for fold_index, validation_indices in enumerate(folds):
            train_mask = torch.ones(len(names), dtype=torch.bool)
            train_mask[validation_indices] = False
            train_indices = torch.arange(len(names))[train_mask]
            torch.manual_seed(seed + fold_index)
            model = MaskedAttentionPhyloRegressor(representations.size(-1), hidden_dim, laplacian)
            predictions, _ = _fit_model(
                model,
                representations,
                position_mask,
                targets_tensor,
                train_indices,
                epochs,
                learning_rate,
                weight_decay,
            )
            validation_targets = targets_tensor[validation_indices]
            denominator = torch.sum((validation_targets - validation_targets.mean()) ** 2)
            numerator = torch.sum((predictions[validation_indices] - validation_targets) ** 2)
            score = 1.0 - numerator / denominator
            if not torch.isfinite(score):
                raise ValueError("Validation fold produced a non-finite R-squared value.")
            fold_scores.append(float(score))

        torch.manual_seed(seed + n_splits)
        final_model = MaskedAttentionPhyloRegressor(representations.size(-1), hidden_dim, laplacian)
        all_indices = torch.arange(len(names))
        _, attention = _fit_model(
            final_model,
            representations,
            position_mask,
            targets_tensor,
            all_indices,
            epochs,
            learning_rate,
            weight_decay,
        )

    detached_attention = attention.detach().clone()
    return RegionAssociationResult(
        cv_r2=sum(fold_scores) / len(fold_scores),
        fold_r2=tuple(fold_scores),
        attention=detached_attention,
        mean_attention=detached_attention.mean(dim=0).detach().clone(),
    )
