"""Leaf-aligned data contracts and preparation entry point."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import torch

from ete3 import Tree

if TYPE_CHECKING:
    from ete3 import Tree


@dataclass(frozen=True)
class LeafRegressionData:
    """Validated, immutable inputs for leaf-level regression.

    Args:
        leaf_names: Nonempty, unique leaf names with shape ``[N]``.
        representations: Float32 position representations ``[N, L, D]``. Models
            validate finite values as they consume raw encoder chunks.
        position_mask: Boolean or strict numeric zero/one valid-position mask
            ``[N, L]`` with at least one valid position per leaf and contiguous
            right padding. Accepted numeric masks are stored as ``torch.bool``.
        targets: Finite float32 targets ``[N]`` in ``leaf_names`` order.
        leaf_laplacian: Finite float32 leaf constraint ``[N, N]`` in the same
            order.

    Tensor fields are validated at construction. ``frozen=True`` prevents field
    reassignment; callers must still treat tensor contents as read-only.
    """

    leaf_names: tuple[str, ...]
    representations: torch.Tensor
    position_mask: torch.Tensor
    targets: torch.Tensor
    leaf_laplacian: torch.Tensor

    def __post_init__(self) -> None:
        position_mask = _normalize_position_mask(self.position_mask)
        _validate_data_fields(
            self.leaf_names,
            self.representations,
            position_mask,
            self.targets,
            self.leaf_laplacian,
        )
        object.__setattr__(self, "position_mask", position_mask)


def _validate_data_fields(
    leaf_names: tuple[str, ...],
    representations: torch.Tensor,
    position_mask: torch.Tensor,
    targets: torch.Tensor,
    leaf_laplacian: torch.Tensor,
) -> None:
    """Validate the canonical fields held by :class:`LeafRegressionData`."""
    if not isinstance(leaf_names, tuple) or not leaf_names:
        raise ValueError("`leaf_names` must be a nonempty tuple of strings.")
    if any(not isinstance(name, str) or not name.strip() for name in leaf_names):
        raise ValueError("`leaf_names` must contain nonblank strings.")
    if len(set(leaf_names)) != len(leaf_names):
        raise ValueError("`leaf_names` must be unique.")

    if not torch.is_tensor(representations) or representations.dtype != torch.float32:
        raise TypeError("`representations` must be a float32 torch.Tensor.")
    if (
        representations.ndim != 3
        or representations.size(0) != len(leaf_names)
        or representations.size(1) == 0
        or representations.size(2) == 0
    ):
        raise ValueError(
            "`representations` must have nonempty shape [N, L, D] aligned to `leaf_names`."
        )
    if position_mask.shape != representations.shape[:2]:
        raise ValueError("`position_mask` must have shape [N, L] matching `representations`.")
    if not torch.all(position_mask.any(dim=1)):
        raise ValueError("Every `position_mask` row must contain a valid position.")
    _validate_contiguous_right_padding(position_mask)

    if not torch.is_tensor(targets) or targets.dtype != torch.float32:
        raise TypeError("`targets` must be a float32 torch.Tensor.")
    if targets.ndim != 1 or targets.numel() != len(leaf_names):
        raise ValueError("`targets` must have shape [N] matching `leaf_names`.")
    if not torch.isfinite(targets).all():
        raise ValueError("`targets` must contain only finite values.")

    if not torch.is_tensor(leaf_laplacian) or leaf_laplacian.dtype != torch.float32:
        raise TypeError("`leaf_laplacian` must be a float32 torch.Tensor.")
    if leaf_laplacian.shape != (len(leaf_names), len(leaf_names)):
        raise ValueError("`leaf_laplacian` must have shape [N, N] matching `leaf_names`.")
    if not torch.isfinite(leaf_laplacian).all():
        raise ValueError("`leaf_laplacian` must contain only finite values.")


def prepare_leaf_regression(
    tree: Tree,
    representations: object,
    position_mask: object,
    targets: Mapping[str, float] | object,
    *,
    leaf_names: Sequence[str] | None = None,
) -> LeafRegressionData:
    """Prepare validated leaf-aligned data.

    Positional targets retain their caller-provided row order. Mapped targets are
    aligned to the final leaf-name order.
    """
    names = _resolve_leaf_names(tree, leaf_names)
    representations_tensor = _as_floating_tensor(representations, "representations")
    position_mask_tensor = _as_position_mask(position_mask)
    if (
        representations_tensor.ndim != 3
        or representations_tensor.size(0) != len(names)
        or representations_tensor.size(1) == 0
        or representations_tensor.size(2) == 0
    ):
        raise ValueError("`representations` must have nonempty shape [N, L, D] aligned to leaves.")
    if position_mask_tensor.shape != representations_tensor.shape[:2]:
        raise ValueError("`position_mask` must have shape [N, L] matching `representations`.")
    if not torch.all(position_mask_tensor.any(dim=1)):
        raise ValueError("Every `position_mask` row must contain a valid position.")
    _validate_contiguous_right_padding(position_mask_tensor)

    return LeafRegressionData(
        leaf_names=names,
        representations=representations_tensor,
        position_mask=position_mask_tensor,
        targets=_as_targets(targets, names),
        leaf_laplacian=_build_leaf_laplacian(tree, names),
    )


def _resolve_leaf_names(tree: Tree, leaf_names: Sequence[str] | None) -> tuple[str, ...]:
    if not isinstance(tree, Tree):
        raise TypeError("`tree` must be an ete3.Tree.")
    tree_names = tuple(leaf.name for leaf in tree.iter_leaves())
    if not tree_names or any(not isinstance(name, str) or not name.strip() for name in tree_names):
        raise ValueError("Tree leaves must have nonblank names.")
    if len(set(tree_names)) != len(tree_names):
        raise ValueError("Tree leaf names must be unique.")
    if leaf_names is None:
        return tree_names
    if isinstance(leaf_names, (str, bytes)) or not isinstance(leaf_names, Sequence):
        raise TypeError("`leaf_names` must be a sequence of strings.")
    names = tuple(leaf_names)
    if (
        len(names) != len(tree_names)
        or any(not isinstance(name, str) or not name.strip() for name in names)
        or len(set(names)) != len(names)
        or set(names) != set(tree_names)
    ):
        raise ValueError("`leaf_names` must be a complete unique permutation of tree leaf names.")
    return names


def _as_floating_tensor(value: object, field_name: str) -> torch.Tensor:
    try:
        tensor = torch.as_tensor(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"`{field_name}` must be tensor-like floating data.") from error
    if not tensor.is_floating_point():
        raise TypeError(f"`{field_name}` must contain floating-point values.")
    return tensor.to(dtype=torch.float32)


def _as_position_mask(value: object) -> torch.Tensor:
    """Convert a boolean or strict numeric zero/one mask to ``torch.bool``."""
    try:
        mask = torch.as_tensor(value)
    except (TypeError, ValueError) as error:
        raise TypeError("`position_mask` must be tensor-like.") from error
    return _normalize_position_mask(mask)


def _normalize_position_mask(mask: torch.Tensor) -> torch.Tensor:
    """Validate a tensor mask and canonicalize accepted numeric values to bool."""
    if not torch.is_tensor(mask):
        raise TypeError("`position_mask` must be a torch.Tensor.")
    if mask.dtype == torch.bool:
        return mask
    if mask.is_complex() or not (mask.is_floating_point() or mask.dtype != torch.bool):
        raise TypeError("`position_mask` must contain boolean or real numeric values.")
    if not torch.isfinite(mask).all():
        raise ValueError("`position_mask` must contain only finite values.")
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("`position_mask` numeric values must be exactly 0 or 1.")
    return mask.to(dtype=torch.bool)


def _validate_contiguous_right_padding(position_mask: torch.Tensor) -> None:
    """Reject position masks that mark a valid token after padding starts."""
    if position_mask.size(1) > 1 and torch.any((~position_mask[:, :-1]) & position_mask[:, 1:]):
        raise ValueError("`position_mask` rows must use contiguous right padding.")


def _as_targets(targets: Mapping[str, float] | object, leaf_names: tuple[str, ...]) -> torch.Tensor:
    if isinstance(targets, Mapping):
        if set(targets) != set(leaf_names):
            raise ValueError("Target mapping keys must exactly match `leaf_names`.")
        result = _as_floating_tensor([targets[name] for name in leaf_names], "targets")
    else:
        result = _as_floating_tensor(targets, "targets")
    if result.ndim != 1 or result.numel() != len(leaf_names):
        raise ValueError("`targets` must have shape [N] matching `leaf_names`.")
    if not torch.isfinite(result).all():
        raise ValueError("`targets` must contain only finite values.")
    return result


def _build_leaf_laplacian(tree: Tree, leaf_names: tuple[str, ...]) -> torch.Tensor:
    leaves = {leaf.name: leaf for leaf in tree.iter_leaves()}
    count = len(leaf_names)
    distances = torch.zeros((count, count), dtype=torch.float32)
    for row in range(count):
        for column in range(row + 1, count):
            distance = float(leaves[leaf_names[row]].get_distance(leaves[leaf_names[column]]))
            if not math.isfinite(distance) or distance < 0:
                raise ValueError("Leaf path distances must be finite and non-negative.")
            distances[row, column] = distance
            distances[column, row] = distance
    if count > 1 and torch.count_nonzero(distances) == 0:
        for row in range(count):
            for column in range(row + 1, count):
                distances[row, column] = distances[column, row] = float(
                    leaves[leaf_names[row]].get_distance(
                        leaves[leaf_names[column]], topology_only=True
                    )
                )
    positive_distances = distances[distances > 0]
    scale = positive_distances.mean() if positive_distances.numel() else torch.tensor(1.0)
    similarity = torch.exp(-distances / scale)
    similarity.fill_diagonal_(0.0)
    degrees = similarity.sum(dim=1)
    inverse_sqrt = torch.where(degrees > 0, degrees.rsqrt(), torch.zeros_like(degrees))
    result = torch.eye(count, dtype=torch.float32) - (
        inverse_sqrt[:, None] * similarity * inverse_sqrt[None, :]
    )
    if not torch.isfinite(result).all():
        raise ValueError("`leaf_laplacian` construction produced non-finite values.")
    return result
