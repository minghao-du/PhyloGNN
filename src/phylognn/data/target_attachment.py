"""Attach externally supplied targets to graph nodes by name."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import copy
import keyword
from typing import Literal

import torch
from torch import Tensor
from torch_geometric.data import Data

_GRAPH_STRUCTURAL_FIELDS = frozenset(
    {
        "x",
        "edge_index",
        "edge_attr",
        "edge_type",
        "face",
        "pos",
        "batch",
        "ptr",
        "adj",
        "adj_t",
        "node_names",
        "num_nodes",
        "num_edges",
        "num_node_features",
        "num_features",
        "num_edge_features",
        "num_faces",
        "original_num_nodes",
        "time_bin",
        "num_time_bins",
        "virtual_node_mask",
        "node_type",
    }
)
_DATA_PUBLIC_API_METHODS = frozenset(
    name for name in dir(Data) if not name.startswith("_") and callable(getattr(Data, name))
)


def attach_node_targets(
    data: Data,
    records: Mapping[str, Mapping[str, object]] | Mapping[str, object],
    *,
    target: str | None = None,
    target_field: str = "y",
    node_selector: Callable[[int, str], bool] | None = None,
    missing: Literal["mask", "error"] = "mask",
    prediction_mask_field: str = "prediction_mask",
    inplace: bool = False,
) -> Data:
    """Attach floating targets and a boolean prediction mask by ``node_names``.

    The returned target has shape ``[N]`` for scalar records or ``[N, D]`` for
    consistently shaped vector records. Unselected and missing nodes receive
    ``NaN`` targets and ``False`` mask values when ``missing="mask"``. The
    default returns a deep copy; ``inplace=True`` updates only the configured
    target and mask fields after all validation succeeds.

    Example:
        >>> from torch_geometric.data import Data
        >>> from phylognn.data import attach_node_targets
        >>> graph = Data(x=torch.ones((2, 1)), node_names=["A", "B"])
        >>> result = attach_node_targets(graph, {"B": 2.0, "A": 1.0})
        >>> result.y.tolist()
        [1.0, 2.0]
    """
    node_names = _validate_graph(data)
    _validate_options(
        target=target,
        target_field=target_field,
        missing=missing,
        prediction_mask_field=prediction_mask_field,
        inplace=inplace,
    )
    record_mode = _validate_records(records, target)
    selected = _select_nodes(node_names, node_selector)
    values, mask, output_shape = _collect_values(
        node_names=node_names,
        records=records,
        record_mode=record_mode,
        target=target,
        selected=selected,
        missing=missing,
    )
    targets = _build_targets(values, output_shape)

    result = data if inplace else copy.deepcopy(data)
    setattr(result, target_field, targets)
    setattr(result, prediction_mask_field, mask)
    return result


def _validate_graph(data: Data) -> list[str]:
    if not isinstance(data, Data):
        raise TypeError(f"data must be a torch_geometric.data.Data, got {type(data).__name__}.")
    if not hasattr(data, "node_names"):
        raise ValueError("data must define node_names for target alignment.")
    node_names = data.node_names
    if isinstance(node_names, (str, bytes)) or not isinstance(node_names, (list, tuple)):
        raise TypeError("data.node_names must be a sequence of strings.")
    if len(node_names) != data.num_nodes:
        raise ValueError(
            "data.node_names length must equal data.num_nodes; "
            f"got {len(node_names)} and {data.num_nodes}."
        )
    for index, name in enumerate(node_names):
        if not isinstance(name, str):
            raise TypeError(
                f"data.node_names[{index}] must be a string, got {type(name).__name__}."
            )
        if not name.strip():
            raise ValueError(f"data.node_names[{index}] must be non-empty and not whitespace-only.")
    if len(set(node_names)) != len(node_names):
        raise ValueError("data.node_names must contain unique names.")
    return list(node_names)


def _validate_options(
    *,
    target: str | None,
    target_field: str,
    missing: str,
    prediction_mask_field: str,
    inplace: bool,
) -> None:
    if target is not None and (not isinstance(target, str) or not target.strip()):
        raise ValueError("target must be a non-empty string when provided.")
    for parameter_name, field_name in (
        ("target_field", target_field),
        ("prediction_mask_field", prediction_mask_field),
    ):
        _validate_output_field_name(parameter_name, field_name)
    if target_field == prediction_mask_field:
        raise ValueError("target_field and prediction_mask_field must be different.")
    if missing not in {"mask", "error"}:
        raise ValueError("missing must be either 'mask' or 'error'.")
    if not isinstance(inplace, bool):
        raise TypeError("inplace must be a bool.")


def _validate_output_field_name(parameter_name: str, field_name: object) -> None:
    if not isinstance(field_name, str):
        raise TypeError(f"{parameter_name} must be a string.")
    if (
        not field_name.strip()
        or not field_name.isidentifier()
        or keyword.iskeyword(field_name)
        or field_name.startswith("_")
    ):
        raise ValueError(
            f"{parameter_name} must be a valid Python attribute name, got {field_name!r}."
        )
    if field_name in _GRAPH_STRUCTURAL_FIELDS:
        raise ValueError(
            f"{parameter_name} cannot overwrite graph-structural field {field_name!r}."
        )
    if field_name in _DATA_PUBLIC_API_METHODS:
        raise ValueError(
            f"{parameter_name} cannot overwrite public Data API method {field_name!r}."
        )


def _validate_records(
    records: Mapping[str, object], target: str | None
) -> Literal["field", "direct"]:
    if not isinstance(records, Mapping):
        raise TypeError(f"records must be a Mapping, got {type(records).__name__}.")
    modes = {"field" if isinstance(value, Mapping) else "direct" for value in records.values()}
    if len(modes) > 1:
        raise ValueError("records must not mix field-record and direct-value forms.")
    mode = next(iter(modes), "field" if target is not None else "direct")
    if mode == "field" and target is None:
        raise ValueError("field-record mappings require a target field name.")
    if mode == "direct" and target is not None:
        raise ValueError("direct-value mappings require target=None.")
    return mode


def _select_nodes(
    node_names: list[str], node_selector: Callable[[int, str], bool] | None
) -> list[bool]:
    if node_selector is None:
        return [True] * len(node_names)
    if not callable(node_selector):
        raise TypeError("node_selector must be callable when provided.")
    selected = []
    for index, name in enumerate(node_names):
        try:
            result = node_selector(index, name)
        except Exception as error:
            raise ValueError(f"node_selector failed for node {index} ({name!r}).") from error
        if not isinstance(result, bool):
            raise TypeError(
                f"node_selector must return bool for node {index} ({name!r}), "
                f"got {type(result).__name__}."
            )
        selected.append(result)
    return selected


def _collect_values(
    *,
    node_names: list[str],
    records: Mapping[str, object],
    record_mode: Literal["field", "direct"],
    target: str | None,
    selected: list[bool],
    missing: Literal["mask", "error"],
) -> tuple[list[Tensor | None], Tensor, tuple[int, ...]]:
    values: list[Tensor | None] = [None] * len(node_names)
    mask = torch.zeros(len(node_names), dtype=torch.bool)
    output_shape: tuple[int, ...] | None = None

    for index, (name, is_selected) in enumerate(zip(node_names, selected, strict=True)):
        if not is_selected:
            continue
        raw_value, available = _get_record_value(records, name, record_mode, target)
        if available:
            value = _convert_value(raw_value, node_name=name, target=target)
            shape = tuple(value.shape)
            if output_shape is None:
                output_shape = shape
            elif shape != output_shape:
                raise ValueError(
                    f"Target shape for node {name!r} is {shape}, expected {output_shape}."
                )
            if torch.isfinite(value).all():
                values[index] = value
                mask[index] = True
                continue
        if missing == "error":
            raise ValueError(f"Target for selected node {name!r} is missing or non-finite.")

    return values, mask, output_shape or ()


def _get_record_value(
    records: Mapping[str, object],
    node_name: str,
    record_mode: Literal["field", "direct"],
    target: str | None,
) -> tuple[object, bool]:
    if node_name not in records:
        return None, False
    record = records[node_name]
    if record_mode == "field":
        assert isinstance(record, Mapping)
        assert target is not None
        if target not in record:
            return None, False
        return record[target], True
    return record, True


def _convert_value(value: object, *, node_name: str, target: str | None) -> Tensor:
    try:
        tensor = torch.as_tensor(value, dtype=torch.float32)
    except (TypeError, ValueError, RuntimeError) as error:
        field_context = f" field {target!r}" if target is not None else ""
        raise TypeError(
            f"Target value for node {node_name!r}{field_context} is not numeric."
        ) from error
    if tensor.ndim > 1:
        raise ValueError(
            f"Target value for node {node_name!r} must be scalar or one-dimensional, "
            f"got shape {tuple(tensor.shape)}."
        )
    return tensor


def _build_targets(values: list[Tensor | None], output_shape: tuple[int, ...]) -> Tensor:
    targets = torch.full((len(values), *output_shape), float("nan"), dtype=torch.float32)
    for index, value in enumerate(values):
        if value is not None:
            targets[index] = value
    return targets
