"""In-memory evaluation of one leaf-aligned region on one phylogenetic tree."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
import math
from numbers import Real
import random
from typing import Iterator

import numpy as np
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


@dataclass(frozen=True)
class RegionAssociationData:
    """Validated, leaf-aligned data for one tree and one continuous region target.

    Fields preserve one final leaf order: ``leaf_names`` has shape ``[N]``,
    representations have shape ``[N, L, D]``, the position mask has shape
    ``[N, L]``, targets have shape ``[N]``, and the leaf constraint has shape
    ``[N, N]``. Tensor fields are float32 except for the boolean mask.
    """

    leaf_names: tuple[str, ...]
    representations: torch.Tensor
    position_mask: torch.Tensor
    targets: torch.Tensor
    leaf_laplacian: torch.Tensor

    def __post_init__(self) -> None:
        _validate_region_association_data_fields(
            self.leaf_names,
            self.representations,
            self.position_mask,
            self.targets,
            self.leaf_laplacian,
        )


@dataclass(frozen=True)
class RegionFitConfig:
    """Immutable validated settings for one region-association fit.

    Args:
        epochs: Positive number of optimization steps, defaulting to ``100``.
        learning_rate: Finite positive Adam learning rate, defaulting to
            ``0.001``.
        weight_decay: Finite non-negative optimizer weight decay, defaulting to
            ``0.0``.
        seed: Optional non-boolean integer for operation-local RNG isolation.
            Seeded CPU fits are bitwise reproducible; non-CPU fits preserve
            caller RNG state without a bitwise reproducibility guarantee.
        device: Optional value accepted by :class:`torch.device`. When set,
            fitting moves fresh model and data copies to that device.
        hidden_dim: Positive hidden width for the default masked-attention
            regressor, defaulting to ``32``.

    Raises:
        ValueError: If a setting is outside its accepted type or value range.
    """

    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    seed: int | None = None
    device: torch.device | str | None = None
    hidden_dim: int = 32

    def __post_init__(self) -> None:
        _validate_positive_int(self.epochs, "epochs")
        _validate_nonnegative_finite_real(self.learning_rate, "learning_rate")
        if self.learning_rate == 0:
            raise ValueError("`learning_rate` must be a positive finite real number.")
        _validate_nonnegative_finite_real(self.weight_decay, "weight_decay")
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise ValueError("`seed` must be a non-boolean integer or None.")
        _validate_device(self.device)
        _validate_positive_int(self.hidden_dim, "hidden_dim")


@dataclass(frozen=True)
class RegionFitResult:
    """Immutable result of one region-association fit.

    ``predictions`` is a detached float tensor of shape ``[N]`` and
    ``attention`` is a detached float tensor of shape ``[N, L]`` with exactly
    zero values at ``~RegionAssociationData.position_mask``. ``train_indices``
    is the validated one-dimensional integer training subset, and ``losses``
    contains one finite Python float for every optimization epoch. Dataclass
    fields cannot be reassigned; tensor contents are not deeply immutable.
    """

    predictions: torch.Tensor
    attention: torch.Tensor
    train_indices: torch.Tensor
    losses: tuple[float, ...]


@dataclass(frozen=True)
class RegionAssociationCVResult:
    """Detached results from one complete leaf-wise cross-validation run.

    ``oof_predictions`` is a finite detached float32 tensor of shape ``[N]``
    populated exactly once per leaf. ``validation_folds`` preserves the caller
    order for manual folds or stores locally generated shuffled folds. Every
    score is a finite Python float, and ``final_fit`` is an all-leaf fit only
    when cross-validation was requested with ``refit=True``.
    """

    cv_score: float
    fold_scores: tuple[float, ...]
    oof_predictions: torch.Tensor
    validation_folds: tuple[torch.Tensor, ...]
    fold_results: tuple[RegionFitResult, ...]
    final_fit: RegionFitResult | None


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


def _validate_region_association_data_fields(
    leaf_names: tuple[str, ...],
    representations: torch.Tensor,
    position_mask: torch.Tensor,
    targets: torch.Tensor,
    leaf_laplacian: torch.Tensor,
) -> None:
    """Validate the already-canonical tensor fields stored by prepared data."""
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
    if not torch.isfinite(representations).all():
        raise ValueError("`representations` must contain only finite values.")
    if not torch.is_tensor(position_mask) or position_mask.dtype != torch.bool:
        raise TypeError("`position_mask` must be a bool torch.Tensor.")
    if position_mask.shape != representations.shape[:2]:
        raise ValueError("`position_mask` must have shape [N, L] matching `representations`.")
    if not torch.all(position_mask.any(dim=1)):
        raise ValueError("Every `position_mask` row must contain a valid position.")
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


def _validate_device(value: torch.device | str | None) -> None:
    """Ensure a requested device is accepted by PyTorch before fitting begins."""
    if value is None:
        return
    try:
        torch.device(value)
    except (TypeError, RuntimeError) as error:
        raise ValueError("`device` must be accepted by torch.device.") from error


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


def _as_leaf_laplacian(value: object, leaf_count: int) -> torch.Tensor:
    """Return a caller-supplied leaf constraint with the public tensor contract."""
    result = _as_floating_tensor(value, "leaf_laplacian")
    if result.shape != (leaf_count, leaf_count):
        raise ValueError(
            "`leaf_laplacian` must have shape [N, N] matching the tree leaf count; "
            f"got {tuple(result.shape)}."
        )
    if not torch.isfinite(result).all():
        raise ValueError("`leaf_laplacian` must contain only finite values.")
    return result


def _validate_region_association_data(data: object) -> RegionAssociationData:
    """Require a prepared data object before fitting or cross-validation."""
    if not isinstance(data, RegionAssociationData):
        raise TypeError("`data` must be a RegionAssociationData instance.")
    return data


def _validate_leaf_indices(
    value: Sequence[int] | torch.Tensor, leaf_count: int, field_name: str
) -> torch.Tensor:
    """Validate a nonempty, unique, in-range one-dimensional leaf index set."""
    try:
        indices = torch.as_tensor(value)
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"`{field_name}` must be a one-dimensional integer index sequence."
        ) from error
    if indices.ndim != 1 or indices.numel() == 0:
        raise ValueError(f"`{field_name}` must be a nonempty one-dimensional index sequence.")
    if indices.dtype == torch.bool or indices.is_floating_point() or indices.is_complex():
        raise TypeError(f"`{field_name}` must contain integer indices.")
    indices = indices.to(dtype=torch.long)
    if torch.any(indices < 0) or torch.any(indices >= leaf_count):
        raise ValueError(f"`{field_name}` contains indices outside [0, {leaf_count}).")
    if torch.unique(indices).numel() != indices.numel():
        raise ValueError(f"`{field_name}` must not contain duplicate indices.")
    return indices


def _validate_validation_folds(value: object, leaf_count: int) -> tuple[torch.Tensor, ...]:
    """Validate ordered manual validation folds before any fitting begins."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("`validation_folds` must be a sequence of index sequences.")
    if len(value) < 2:
        raise ValueError("`validation_folds` must contain at least two nonempty folds.")

    folds = tuple(
        _validate_leaf_indices(fold, leaf_count, f"validation_folds[{index}]").cpu()
        for index, fold in enumerate(value)
    )
    assigned_indices = torch.cat(folds)
    if (
        assigned_indices.numel() != leaf_count
        or torch.unique(assigned_indices).numel() != leaf_count
    ):
        raise ValueError("`validation_folds` must assign every leaf exactly once without overlap.")
    if not torch.equal(torch.sort(assigned_indices).values, torch.arange(leaf_count)):
        raise ValueError("`validation_folds` must cover every leaf index in [0, N) exactly once.")
    return tuple(_detached_clone(fold) for fold in folds)


def _generate_validation_folds(
    leaf_count: int, n_splits: object, seed: int | None
) -> tuple[torch.Tensor, ...]:
    """Generate local shuffled K-folds without changing caller RNG state."""
    if isinstance(n_splits, bool) or not isinstance(n_splits, int):
        raise ValueError("`n_splits` must be an integer.")
    if n_splits < 2 or n_splits > leaf_count // 2:
        raise ValueError("`n_splits` must satisfy 2 <= n_splits <= floor(N / 2).")

    generator = torch.Generator()
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(seed)
    return tuple(torch.tensor_split(torch.randperm(leaf_count, generator=generator), n_splits))


def _default_r2_score(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculate one fold's R-squared, rejecting undefined constant targets."""
    denominator = torch.sum((targets - targets.mean()) ** 2)
    if denominator.item() == 0:
        raise ValueError("Validation fold has constant targets, so R-squared is undefined.")
    score = 1.0 - torch.sum((predictions - targets) ** 2) / denominator
    return _validate_score_value(score, "default R-squared")


def _validate_score_value(value: object, name: str) -> float:
    """Convert one finite detached scalar score to a Python float."""
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(f"`{name}` must return a one-element scalar score tensor.")
        if value.requires_grad:
            raise ValueError(
                f"`{name}` must return a score tensor that does not require gradients."
            )
        if value.is_complex():
            raise TypeError(f"`{name}` must return a real-valued score tensor.")
        score = float(value.detach().cpu())
    else:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"`{name}` must return a numeric scalar score.")
        score = float(value)
    if not math.isfinite(score):
        raise ValueError(f"`{name}` returned a non-finite score.")
    return score


def _validate_factory_result(value: object, expected_type: type[object], factory_name: str) -> None:
    """Require a custom factory to produce the expected direct-call result type."""
    if not isinstance(value, expected_type):
        raise TypeError(
            f"`{factory_name}` returned {type(value).__name__}, not {expected_type.__name__}."
        )


def _validate_model_output(
    output: object, leaf_count: int, position_count: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate custom model output before an optimizer step can occur."""
    if not isinstance(output, tuple) or len(output) != 2:
        raise TypeError("The model must return a `(predictions, attention)` tuple.")
    predictions, attention = output
    if not torch.is_tensor(predictions) or not torch.is_tensor(attention):
        raise TypeError("Model predictions and attention must be torch.Tensor instances.")
    if predictions.shape != (leaf_count,):
        raise ValueError("Model predictions must have shape [N].")
    if attention.shape != (leaf_count, position_count):
        raise ValueError("Model attention must have shape [N, L].")
    if not predictions.is_floating_point() or not attention.is_floating_point():
        raise TypeError("Model predictions and attention must have floating-point dtypes.")
    if not torch.isfinite(predictions).all() or not torch.isfinite(attention).all():
        raise ValueError("Model predictions and attention must contain only finite values.")
    return predictions, attention


def _validate_training_loss(loss: object) -> torch.Tensor:
    """Require a differentiable one-element tensor loss before optimizer stepping."""
    if not torch.is_tensor(loss):
        raise TypeError("The loss function must return a torch.Tensor.")
    if loss.numel() != 1:
        raise ValueError("The loss function must return a one-element tensor.")
    if not loss.is_floating_point() or not loss.requires_grad:
        raise ValueError("The loss function must return a differentiable floating-point tensor.")
    if not torch.isfinite(loss).all():
        raise ValueError("The loss function returned a non-finite value.")
    return loss


def _validate_optimizer_parameter_coverage(
    optimizer: object, model: torch.nn.Module
) -> torch.optim.Optimizer:
    """Ensure an optimizer is valid and owns every trainable model parameter."""
    _validate_factory_result(optimizer, torch.optim.Optimizer, "optimizer_factory")
    optimizer_parameter_ids = {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }
    missing_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in optimizer_parameter_ids
    ]
    if missing_parameters:
        raise ValueError("The optimizer must include every trainable model parameter.")
    return optimizer


def _detached_clone(value: torch.Tensor) -> torch.Tensor:
    """Return an independent tensor that cannot retain training graph state."""
    return value.detach().clone()


@contextmanager
def _local_seed(seed: int | None, device: torch.device | str | None) -> Iterator[None]:
    """Isolate Python, NumPy, and PyTorch RNG changes made by one fit operation."""
    resolved_device = torch.device(device) if device is not None else torch.device("cpu")
    cuda_devices: list[int] = []
    if resolved_device.type == "cuda" and torch.cuda.is_available():
        cuda_devices.append(
            resolved_device.index
            if resolved_device.index is not None
            else torch.cuda.current_device()
        )
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    with torch.random.fork_rng(devices=cuda_devices):
        try:
            if seed is not None:
                torch.manual_seed(seed)
                random.seed(seed)
                np.random.seed(seed % (2**32))
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


def prepare_region_association(
    tree: Tree,
    representations: object,
    position_mask: object,
    targets: Mapping[str, float] | object,
    *,
    leaf_names: Sequence[str] | None = None,
    leaf_laplacian: object | None = None,
) -> RegionAssociationData:
    """Prepare strictly validated, leaf-aligned inputs for region association.

    Args:
        tree: An ETE tree with unique, nonblank leaf names.
        representations: Floating values with nonempty shape ``[N, L, D]``.
        position_mask: Bool-convertible values with shape ``[N, L]`` and at
            least one valid position for every leaf.
        targets: Floating ``[N]`` values already in final leaf order, or a
            mapping whose keys exactly match the final leaf names.
        leaf_names: An optional complete, unique permutation of the tree's leaf
            names. It defines the returned order but never reorders raw tensors.
        leaf_laplacian: An optional finite, floating ``[N, N]`` leaf constraint.
            Its Laplacian semantics are not inferred or validated.

    Returns:
        Frozen prepared data with float32 representations, targets, and leaf
        constraint, plus a bool position mask in the requested leaf order.

    Raises:
        TypeError: If the tree or raw values do not meet their required types.
        ValueError: If names, shapes, finite values, masks, targets, or the leaf
            constraint violate the public preparation contract.
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
        raise ValueError(
            "`representations` must have nonempty shape [N, L, D] aligned to tree leaves; "
            f"got {tuple(representations_tensor.shape)}."
        )
    if not torch.isfinite(representations_tensor).all():
        raise ValueError("`representations` must contain only finite values.")
    if position_mask_tensor.shape != representations_tensor.shape[:2]:
        raise ValueError(
            "`position_mask` must have shape [N, L] matching `representations`; "
            f"got {tuple(position_mask_tensor.shape)}."
        )
    if not torch.all(position_mask_tensor.any(dim=1)):
        raise ValueError("Every `position_mask` row must contain a valid position.")
    targets_tensor = _as_targets(targets, names)
    constraint = (
        build_leaf_laplacian(tree, names)
        if leaf_laplacian is None
        else _as_leaf_laplacian(leaf_laplacian, len(names))
    )
    return RegionAssociationData(
        leaf_names=names,
        representations=representations_tensor,
        position_mask=position_mask_tensor,
        targets=targets_tensor,
        leaf_laplacian=constraint,
    )


def fit_region_association(
    data: RegionAssociationData,
    *,
    train_indices: Sequence[int] | torch.Tensor | None = None,
    config: RegionFitConfig | None = None,
    model_factory: Callable[[], torch.nn.Module] | None = None,
    loss_factory: Callable[[], Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] | None = None,
    optimizer_factory: (
        Callable[[Iterable[torch.nn.Parameter]], torch.optim.Optimizer] | None
    ) = None,
) -> RegionFitResult:
    """Fit one masked-attention regressor on selected leaves of prepared data.

    Args:
        data: Validated leaf-aligned data from :func:`prepare_region_association`.
        train_indices: Optional nonempty, unique ``[K]`` integer leaf indices.
            When omitted, every leaf is used for the training loss.
        config: Frozen training settings. Defaults to :class:`RegionFitConfig`.
            A requested device receives fresh model and tensor copies.
        model_factory: Optional zero-argument factory returning a
            :class:`torch.nn.Module`. The default builds
            :class:`MaskedAttentionPhyloRegressor`.
        loss_factory: Optional zero-argument factory returning a callable that
            accepts selected predictions and targets. The default is MSE loss.
        optimizer_factory: Optional factory accepting ``model.parameters()``.
            The default is Adam configured from ``config``.

    Returns:
        A frozen result with detached-clone all-leaf float predictions ``[N]``,
        attention ``[N, L]`` that is zero at padding positions, the validated
        training indices, and one finite loss value per epoch.

    Raises:
        TypeError: If prepared data, configuration, factories, or factory
            results violate their contracts.
        ValueError: If indices, model outputs, losses, optimizer coverage, or
            non-finite training values violate the fit contract. These checks
            complete before the first optimizer step.

    Factories are invoked directly as ``model_factory()``, ``loss_factory()``,
    and ``optimizer_factory(model.parameters())``. When ``config.seed`` is set,
    Python, NumPy, and PyTorch RNG state is restored before this function
    returns; CPU default fits are bitwise reproducible for equal inputs.
    """
    prepared = _validate_region_association_data(data)
    if config is None:
        config = RegionFitConfig()
    elif not isinstance(config, RegionFitConfig):
        raise TypeError("`config` must be a RegionFitConfig instance or None.")

    leaf_count = len(prepared.leaf_names)
    if train_indices is None:
        selected_indices = torch.arange(leaf_count, dtype=torch.long)
    else:
        selected_indices = _validate_leaf_indices(train_indices, leaf_count, "train_indices")

    if config.device is None:
        device = prepared.representations.device
    else:
        device = torch.device(config.device)

    with _local_seed(config.seed, device):
        representations = _detached_clone(prepared.representations).to(device)
        position_mask = _detached_clone(prepared.position_mask).to(device)
        targets = _detached_clone(prepared.targets).to(device)
        selected_indices = selected_indices.to(device)

        if model_factory is None:
            model = MaskedAttentionPhyloRegressor(
                representations.size(-1),
                config.hidden_dim,
                _detached_clone(prepared.leaf_laplacian).to(device),
            )
        else:
            if not callable(model_factory):
                raise TypeError("`model_factory` must be callable.")
            model = model_factory()
            _validate_factory_result(model, torch.nn.Module, "model_factory")
        model = model.to(device)

        if loss_factory is None:
            loss_function = torch.nn.functional.mse_loss
        else:
            if not callable(loss_factory):
                raise TypeError("`loss_factory` must be callable.")
            loss_function = loss_factory()
            if not callable(loss_function):
                raise TypeError("`loss_factory` must return a callable loss function.")

        if optimizer_factory is None:
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        else:
            if not callable(optimizer_factory):
                raise TypeError("`optimizer_factory` must be callable.")
            optimizer = optimizer_factory(model.parameters())
        optimizer = _validate_optimizer_parameter_coverage(optimizer, model)

        losses: list[float] = []
        model.train()
        for _ in range(config.epochs):
            optimizer.zero_grad()
            predictions, attention = _validate_model_output(
                model(representations, position_mask),
                leaf_count,
                representations.size(1),
            )
            loss = _validate_training_loss(
                loss_function(predictions[selected_indices], targets[selected_indices])
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            predictions, attention = _validate_model_output(
                model(representations, position_mask),
                leaf_count,
                representations.size(1),
            )
            attention = attention * position_mask.to(dtype=attention.dtype)

    return RegionFitResult(
        predictions=_detached_clone(predictions),
        attention=_detached_clone(attention),
        train_indices=_detached_clone(selected_indices),
        losses=tuple(losses),
    )


def cross_validate_region_association(
    data: RegionAssociationData,
    *,
    n_splits: int = 5,
    validation_folds: Sequence[Sequence[int] | torch.Tensor] | None = None,
    config: RegionFitConfig | None = None,
    score_fn: Callable[[torch.Tensor, torch.Tensor], float | torch.Tensor] | None = None,
    refit: bool = True,
    model_factory: Callable[[], torch.nn.Module] | None = None,
    loss_factory: Callable[[], Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] | None = None,
    optimizer_factory: (
        Callable[[Iterable[torch.nn.Parameter]], torch.optim.Optimizer] | None
    ) = None,
) -> RegionAssociationCVResult:
    """Cross-validate prepared region data and optionally refit every leaf.

    Args:
        data: Validated leaf-aligned input data.
        n_splits: Number of generated shuffled validation folds. When manual
            folds are absent it must satisfy ``2 <= n_splits <= floor(N / 2)``.
        validation_folds: Optional ordered, nonempty validation index sets.
            They must cover each leaf exactly once and retain their input order.
        config: Training settings shared by each fit. A seeded CPU run has
            bitwise-repeatable generated folds, fits, scores, and OOF output.
        score_fn: Optional scorer receiving one fold's predictions and targets.
            It must return a finite Python numeric scalar or detached one-element
            real-valued tensor. The default is per-fold R-squared.
        refit: Whether to fit once more over every leaf after cross-validation.
        model_factory: Optional model factory passed directly to each fit.
        loss_factory: Optional loss factory passed directly to each fit.
        optimizer_factory: Optional optimizer factory passed directly to each fit.

    Returns:
        Frozen scores, complete detached OOF predictions of shape ``[N]``,
        ordered validation folds, one result per fold, and an optional all-leaf
        ``final_fit``.

    Raises:
        TypeError: If data, configuration, folds, or a custom scorer violates
            its type contract.
        ValueError: If folds do not cover every leaf once, generated split
            counts are invalid, default R-squared is undefined, or a scorer
            returns an invalid score. Fold validation completes before fitting.

    Manual folds are used verbatim after validation. Generated folds use a
    local generator; when ``config.seed`` is set, all Python, NumPy, and
    PyTorch caller RNG state is restored before return. Non-CPU devices retain
    RNG isolation without a bitwise reproducibility guarantee.
    """
    prepared = _validate_region_association_data(data)
    if config is None:
        config = RegionFitConfig()
    elif not isinstance(config, RegionFitConfig):
        raise TypeError("`config` must be a RegionFitConfig instance or None.")
    if not isinstance(refit, bool):
        raise TypeError("`refit` must be a bool.")
    if score_fn is not None and not callable(score_fn):
        raise TypeError("`score_fn` must be callable or None.")

    leaf_count = len(prepared.leaf_names)
    if validation_folds is None:
        folds = _generate_validation_folds(leaf_count, n_splits, config.seed)
    else:
        folds = _validate_validation_folds(validation_folds, leaf_count)

    if score_fn is None:
        for validation_indices in folds:
            _default_r2_score(
                prepared.targets[validation_indices], prepared.targets[validation_indices]
            )

    device = torch.device(config.device) if config.device is not None else prepared.targets.device
    all_indices = torch.arange(leaf_count, dtype=torch.long)
    fold_scores: list[float] = []
    fold_results: list[RegionFitResult] = []
    oof_predictions = torch.empty(leaf_count, dtype=torch.float32, device=device)
    with _local_seed(config.seed, device):
        for fold_index, validation_indices in enumerate(folds):
            train_indices = all_indices[~torch.isin(all_indices, validation_indices)]
            fit_config = (
                config if config.seed is None else replace(config, seed=config.seed + fold_index)
            )
            result = fit_region_association(
                prepared,
                train_indices=train_indices,
                config=fit_config,
                model_factory=model_factory,
                loss_factory=loss_factory,
                optimizer_factory=optimizer_factory,
            )
            validation_indices = validation_indices.to(result.predictions.device)
            predictions = result.predictions[validation_indices]
            targets = prepared.targets.to(result.predictions.device)[validation_indices]
            score = (
                _default_r2_score(predictions, targets)
                if score_fn is None
                else _validate_score_value(score_fn(predictions, targets), "score_fn")
            )
            oof_predictions[validation_indices] = predictions
            fold_scores.append(score)
            fold_results.append(result)

        final_fit = None
        if refit:
            final_config = (
                config if config.seed is None else replace(config, seed=config.seed + len(folds))
            )
            final_fit = fit_region_association(
                prepared,
                train_indices=all_indices,
                config=final_config,
                model_factory=model_factory,
                loss_factory=loss_factory,
                optimizer_factory=optimizer_factory,
            )

    return RegionAssociationCVResult(
        cv_score=sum(fold_scores) / len(fold_scores),
        fold_scores=tuple(fold_scores),
        oof_predictions=_detached_clone(oof_predictions),
        validation_folds=tuple(_detached_clone(fold) for fold in folds),
        fold_results=tuple(fold_results),
        final_fit=final_fit,
    )


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
    """Run the original one-shot workflow through preparation and CV stages.

    The public signature and :class:`RegionAssociationResult` fields remain
    unchanged. Preparation owns alignment and validation; cross-validation
    owns folds, scoring, and exactly one all-leaf final fit whose detached
    attention is reused here.
    """
    prepared = prepare_region_association(
        tree,
        representations,
        position_mask,
        targets,
        leaf_names=leaf_names,
    )
    config = RegionFitConfig(
        epochs=epochs,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
    )
    cv_result = cross_validate_region_association(
        prepared,
        n_splits=n_splits,
        config=config,
        refit=True,
    )
    if cv_result.final_fit is None:
        raise RuntimeError("Cross-validation did not return the required final fit.")
    detached_attention = _detached_clone(cv_result.final_fit.attention)
    return RegionAssociationResult(
        cv_r2=cv_result.cv_score,
        fold_r2=cv_result.fold_scores,
        attention=detached_attention,
        mean_attention=detached_attention.mean(dim=0).detach().clone(),
    )
