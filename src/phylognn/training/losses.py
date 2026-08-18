"""Parameter-aware shared loss catalog for leaf regression and TOML training.

This module provides the single source of truth for supported loss functions,
their parameters, validation, construction, and identifier formatting.  It
depends only on :mod:`torch.nn` so that both the programmatic leaf-regression
surface and the TOML-driven training path can import it without pulling in the
model registry or the configuration loader.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Mapping
from numbers import Real
from types import MappingProxyType
from typing import NamedTuple

import torch
import torch.nn as nn


class PGLSLoss(nn.Module):
    """Differentiable per-tree phylogenetic generalized least-squares loss.

    ``predictions`` and ``targets`` are ``[N, T]`` tensors (targets may be
    ``[N]`` for one trait). ``covariances[i]`` belongs to leaves whose
    ``batch`` identifier is ``i``. For each tree the GLS quadratic form is
    normalized by leaf count and averaged over traits, then tree losses are
    arithmetically averaged. Covariances and tensors must share dtype/device.

    Returns
    -------
    torch.Tensor
        A differentiable scalar loss in the prediction dtype and device.

    Raises
    ------
    TypeError
        If a tensor object or covariance container is invalid.
    ValueError
        If a tensor shape, dtype, device, tree mapping, or value is invalid.
    """

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        covariances: list[torch.Tensor],
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the scalar objective while preserving prediction gradients.

        Parameters follow the class-level ``[N, T]`` prediction/target,
        per-tree covariance-list, and ``[N]`` batch mapping contract.
        """
        if not torch.is_tensor(predictions):
            raise TypeError("predictions must be a torch.Tensor.")
        if not torch.is_tensor(targets):
            raise TypeError("targets must be a torch.Tensor.")
        if not torch.is_tensor(batch):
            raise TypeError("batch must be a torch.Tensor.")
        if not isinstance(covariances, list):
            raise TypeError("covariances must be a list of torch.Tensor objects.")
        if any(not torch.is_tensor(covariance) for covariance in covariances):
            raise TypeError("each covariance must be a torch.Tensor.")

        if predictions.ndim != 2:
            raise ValueError("predictions must have shape [N, T].")
        if targets.ndim == 1 and predictions.shape[1] == 1:
            targets = targets.unsqueeze(1)
        elif targets.ndim != 2:
            raise ValueError("targets must have shape [N, T].")
        if batch.ndim != 1:
            raise ValueError("batch must have shape [N].")
        if predictions.shape[0] == 0 or predictions.shape[1] == 0:
            raise ValueError("predictions and targets must be non-empty.")
        if predictions.shape != targets.shape:
            raise ValueError("predictions and targets must have matching shapes.")
        if batch.shape[0] != predictions.shape[0]:
            raise ValueError("batch must have shape [N] matching predictions.")
        if not covariances:
            raise ValueError("covariances must be a non-empty list.")

        if predictions.dtype not in (torch.float32, torch.float64):
            raise ValueError("predictions dtype must be torch.float32 or torch.float64.")
        if targets.dtype != predictions.dtype:
            raise ValueError("targets must use the same dtype as predictions.")
        if batch.dtype != torch.long:
            raise ValueError("batch dtype must be torch.int64.")
        if targets.device != predictions.device:
            raise ValueError("targets device must match the predictions device.")
        if batch.device != predictions.device:
            raise ValueError("batch device must match the predictions device.")
        if not torch.isfinite(predictions).all():
            raise ValueError("predictions must contain only finite values.")
        if not torch.isfinite(targets).all():
            raise ValueError("targets must contain only finite values.")

        represented_trees = torch.unique(batch, sorted=True)
        if represented_trees[0].item() < 0:
            raise ValueError("batch identifiers must be non-negative.")
        expected_trees = torch.arange(
            represented_trees.numel(), dtype=torch.long, device=batch.device
        )
        if not torch.equal(represented_trees, expected_trees):
            raise ValueError("batch identifiers must be contiguous and cover 0..K-1.")
        if represented_trees.numel() != len(covariances):
            raise ValueError("covariance count must match the represented batch tree count.")

        tree_indices: list[torch.Tensor] = []
        for tree_id, covariance in enumerate(covariances):
            indices = torch.where(batch == tree_id)[0]
            if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
                raise ValueError(f"covariance {tree_id} must be a square matrix.")
            if covariance.shape[0] != indices.numel():
                raise ValueError(f"covariance {tree_id} shape must match its tree leaf count.")
            if covariance.dtype != predictions.dtype:
                raise ValueError("each covariance must use the same dtype as predictions.")
            if covariance.device != predictions.device:
                raise ValueError("covariances device must match the predictions device.")
            if not torch.isfinite(covariance).all():
                raise ValueError("covariances must contain only finite values.")
            if not torch.allclose(covariance, covariance.T, rtol=1e-5, atol=1e-6):
                raise ValueError(f"covariance {tree_id} must be symmetric.")
            eigenvalues = torch.linalg.eigvalsh(covariance)
            minimum = eigenvalues[0]
            maximum = eigenvalues[-1]
            if minimum.item() <= 0 or maximum.item() <= 0:
                raise ValueError(f"covariance {tree_id} must be positive definite.")
            if (minimum / maximum).item() < 1e-6:
                raise ValueError(
                    f"covariance {tree_id} condition ratio must be greater than or equal to 1e-6."
                )
            tree_indices.append(indices)

        tree_losses: list[torch.Tensor] = []
        for covariance, indices in zip(covariances, tree_indices):
            residual = targets.index_select(0, indices) - predictions.index_select(0, indices)
            solved = torch.linalg.solve(covariance, residual)
            tree_losses.append((residual * solved).sum(dim=0).mean() / residual.shape[0])
        return torch.stack(tree_losses).mean()


# ---------------------------------------------------------------------------
# Catalog entry
# ---------------------------------------------------------------------------


class _LossSpec(NamedTuple):
    """Immutable specification of a single supported loss.

    Attributes
    ----------
    factory : type[nn.Module]
        Loss class, constructed with mean reduction and resolved parameters.
    parameter_names : frozenset[str]
        Accepted parameter names. Empty for parameter-free losses.
    defaults : Mapping[str, float]
        Default value per parameter, applied when the caller omits it.
    """

    factory: type[nn.Module]
    parameter_names: frozenset[str] = frozenset()
    defaults: Mapping[str, float] = MappingProxyType({})


# ---------------------------------------------------------------------------
# Private catalog
# ---------------------------------------------------------------------------

_LOSS_CATALOG: dict[str, _LossSpec] = {
    "mse": _LossSpec(factory=nn.MSELoss),
    "mae": _LossSpec(factory=nn.L1Loss),
    "huber": _LossSpec(
        factory=nn.HuberLoss,
        parameter_names=frozenset({"delta"}),
        defaults=MappingProxyType({"delta": 1.0}),
    ),
}


# ---------------------------------------------------------------------------
# Public query
# ---------------------------------------------------------------------------


def supported_loss_names() -> tuple[str, ...]:
    """Return the supported loss identifiers in sorted order.

    A new tuple is returned on each call so callers cannot mutate catalog
    state.  This is the public discovery entry point for loss selection.

    Returns
    -------
    tuple[str, ...]
        Sorted tuple of supported loss name strings.

    Examples
    --------
    >>> from phylognn.training.losses import supported_loss_names
    >>> supported_loss_names()
    ('huber', 'mae', 'mse')
    """
    return tuple(sorted(_LOSS_CATALOG))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _default_error_factory(
    message: str, category: type[Exception], *, rejection: str = ""
) -> Exception:
    return category(message)


def resolve_loss_selection(
    name: object,
    params: Mapping[str, object] | None,
    *,
    error_factory: Callable[..., Exception] | None = None,
) -> tuple[str, dict[str, float]]:
    """Validate a loss name and its parameters, returning the resolved pair.

    This is the single validation body shared by every configuration surface.
    The ``error_factory`` callback controls how the category of each rejection
    (``TypeError`` for wrong-typed values, ``ValueError`` for unsupported or
    out-of-range values) is turned into a raised exception.

    Parameters
    ----------
    name : object
        Loss identifier to validate.  Must be a ``str`` present in the
        catalog.
    params : Mapping[str, object] | None
        User-supplied parameters, or ``None`` to accept defaults.
    error_factory : Callable[..., Exception] | None
        Factory ``(message, category, *, rejection) -> Exception``.  The
        ``rejection`` keyword is one of ``"name"`` (loss name validation),
        ``"params"`` (parameter-set-level rejection such as parameter-free
        or unknown key), or ``"param_value"`` (individual parameter value
        validation).  Defaults to
        ``lambda message, category, **_: category(message)``.

    Returns
    -------
    tuple[str, dict[str, float]]
        ``(resolved_name, resolved_params)`` with defaults applied and
        numeric values normalized to ``float``.

    Raises
    ------
    TypeError
        If *name* is not a string, or a parameter value has the wrong type
        (via the default error factory).
    ValueError
        If *name* is unsupported, or a parameter is out of range (via the
        default error factory).
    """
    if error_factory is None:
        error_factory = _default_error_factory
    else:
        _raw_factory = error_factory
        try:
            inspect.signature(_raw_factory).bind("message", ValueError, rejection="param_value")
        except TypeError:
            accepts_rejection = False
        except ValueError:
            accepts_rejection = False
        else:
            accepts_rejection = True

        def error_factory(
            message: str, category: type[Exception], *, rejection: str = ""
        ) -> Exception:
            if accepts_rejection:
                return _raw_factory(message, category, rejection=rejection)
            return _raw_factory(message, category)

    # --- name validation ---
    if not isinstance(name, str):
        raise error_factory(
            f"loss name must be a string, got {type(name).__name__}.",
            TypeError,
            rejection="name",
        )

    if name not in _LOSS_CATALOG:
        supported = ", ".join(supported_loss_names())
        raise error_factory(
            f"unsupported loss {name!r}; supported names are: {supported}.",
            ValueError,
            rejection="name",
        )

    spec = _LOSS_CATALOG[name]
    supplied = dict(params) if params else {}

    # Parameter names are catalog identifiers, so reject non-string keys before
    # the formatting and membership checks below can produce incidental errors.
    non_string_keys = [key for key in supplied if not isinstance(key, str)]
    if non_string_keys:
        bad = ", ".join(repr(key) for key in sorted(non_string_keys, key=repr))
        raise error_factory(
            f"unknown parameter(s) {bad} for loss {name!r}; parameter names must be strings.",
            ValueError,
            rejection="params",
        )

    # --- reject parameters for parameter-free losses ---
    if not spec.parameter_names and supplied:
        keys = ", ".join(sorted(supplied))
        # Build an actionable hint naming which losses accept each supplied key.
        hints: list[str] = []
        for key in sorted(supplied):
            owners = sorted(n for n, s in _LOSS_CATALOG.items() if key in s.parameter_names)
            if owners:
                hints.append(f"{key} is accepted by: {', '.join(owners)}")
        hint_text = "; ".join(hints)
        suffix = f" ({hint_text})" if hint_text else ""
        raise error_factory(
            f"loss {name!r} does not accept parameters, got: {keys}.{suffix}",
            ValueError,
            rejection="params",
        )

    # --- reject unknown parameter names ---
    unknown = set(supplied) - spec.parameter_names
    if unknown:
        bad = ", ".join(sorted(unknown))
        accepted = ", ".join(sorted(spec.parameter_names))
        raise error_factory(
            f"unknown parameter(s) {bad} for loss {name!r}; "
            f"accepted parameters are: {accepted}.",
            ValueError,
            rejection="params",
        )

    # --- validate and normalize each supplied parameter value ---
    resolved: dict[str, float] = {}
    for pname in sorted(spec.parameter_names):
        if pname in supplied:
            value = supplied[pname]
            _validate_parameter_value(pname, value, name, error_factory)
            resolved[pname] = float(value)  # type: ignore[arg-type]
        elif pname in spec.defaults:
            resolved[pname] = spec.defaults[pname]

    return name, resolved


def _validate_parameter_value(
    pname: str,
    value: object,
    loss_name: str,
    error_factory: Callable[..., Exception],
) -> None:
    """Validate a single parameter value for type, finiteness, and range."""
    # bool must be rejected before the numeric check because bool is a
    # subclass of int in Python.
    if isinstance(value, bool):
        raise error_factory(
            f"{pname} for loss {loss_name!r} must be a real number, "
            f"got {type(value).__name__}.",
            TypeError,
            rejection="param_value",
        )
    if not isinstance(value, Real):
        raise error_factory(
            f"{pname} for loss {loss_name!r} must be a real number, "
            f"got {type(value).__name__}.",
            TypeError,
            rejection="param_value",
        )
    fval = float(value)
    if not math.isfinite(fval):
        raise error_factory(
            f"{pname} for loss {loss_name!r} must be finite, got {value!r}.",
            ValueError,
            rejection="param_value",
        )
    if fval <= 0:
        raise error_factory(
            f"{pname} for loss {loss_name!r} must be strictly positive, " f"got {value!r}.",
            ValueError,
            rejection="param_value",
        )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def build_loss(name: str, params: Mapping[str, float]) -> nn.Module:
    """Construct the loss module for an already-resolved selection.

    The returned module uses mean reduction and the resolved parameters as
    keyword arguments.

    Parameters
    ----------
    name : str
        A catalog loss name (must already be validated).
    params : Mapping[str, float]
        Fully resolved parameter mapping from :func:`resolve_loss_selection`.

    Returns
    -------
    torch.nn.Module
        The constructed loss module.
    """
    spec = _LOSS_CATALOG[name]
    return spec.factory(reduction="mean", **params)


# ---------------------------------------------------------------------------
# Identifier formatting
# ---------------------------------------------------------------------------


def format_loss_identifier(name: str, params: Mapping[str, float]) -> str:
    """Render the canonical tracking identifier for a resolved selection.

    Parameter-free losses return the bare name.  Parameterized losses render
    as ``name(param=value)`` with each value formatted as ``repr(float(value))``
    and parameters in sorted name order.

    Parameters
    ----------
    name : str
        Catalog loss name.
    params : Mapping[str, float]
        Resolved parameter mapping.

    Returns
    -------
    str
        Canonical loss identifier string.

    Examples
    --------
    >>> from phylognn.training.losses import format_loss_identifier
    >>> format_loss_identifier("mse", {})
    'mse'
    >>> format_loss_identifier("huber", {"delta": 1.0})
    'huber(delta=1.0)'
    """
    if not params:
        return name
    param_strs = ", ".join(f"{k}={float(v)!r}" for k, v in sorted(params.items()))
    return f"{name}({param_strs})"
