"""
Split-aware dataset utilities for phylogenetic graph data.

This module provides industrialized dataset implementations for PyTorch Geometric
with explicit support for dataset splits such as train / val / test.

Overview
--------
The module contains two primary dataset classes:

1. SplitPhyloDataset
   - In-memory dataset
   - Use when all graphs are already loaded as `torch_geometric.data.Data`

2. SplitPhyloDiskDataset
   - Disk-backed dataset
   - Use when graphs (`x`) and labels (`y`) are stored as `.pt` files on disk

In addition, this module provides:
- split generation utilities
- deterministic random splitting
- split loading from explicit sample IDs or manifest files
- subset/view creation for split-specific datasets
- strong validation and consistent target attachment

Core design principles
----------------------
- Deterministic indexing and splitting
- Clear and stable matching rules
- Strong input validation
- Separation of concerns:
    * indexing
    * split construction
    * graph loading
    * label loading
    * target attachment
- Production-friendly APIs
- Safe cloning before mutation
- PyTorch Geometric compatibility

Recommended on-disk layout
--------------------------
Use mirrored graph/label filenames:

    dataset/
        graphs/
            sample_0001.pt
            sample_0002.pt
            ...
        labels/
            sample_0001.pt
            sample_0002.pt
            ...

Optional split manifests:

    dataset/
        splits/
            train.txt
            val.txt
            test.txt

Where each split file contains one sample ID per line, for example:
    sample_0001
    sample_0002
    sample_0042

Definition of sample ID
-----------------------
By default, the sample ID of a graph file is its relative path to `graph_dir`
without the `.pt` suffix.

Examples:
- graph path: `graphs/a.pt`                -> sample_id = "a"
- graph path: `graphs/fold1/a.pt`          -> sample_id = "fold1/a"

This convention enables:
- deterministic matching
- mirrored label lookup
- split manifests independent of absolute paths

Label attachment contract
-------------------------
Single-task:
- `data.y`

Multi-task:
- `data.y_<task_name>` for each task
- `data.task_names` as a list of original task names

Rationale:
- attaching a Python dict directly to `data.y` is often less robust for PyG
  batching and downstream engineering than explicit attributes.

Accepted label formats
----------------------
Each label object may be:
- `torch.Tensor`
- numeric scalar (`int`, `float`, `bool`)
- `dict[str, torch.Tensor | numeric scalar]`

Split strategies
----------------
This module supports two split strategies:

1. Ratio-based random split
   Example:
       split = DatasetSplit.from_ratios(
           sample_ids=dataset.sample_ids,
           train_ratio=0.8,
           val_ratio=0.1,
           test_ratio=0.1,
           seed=42,
       )

2. Explicit split definition
   Example:
       split = DatasetSplit.from_dict({
           "train": ["sample_0001", "sample_0002"],
           "val":   ["sample_0003"],
           "test":  ["sample_0004"],
       })

Manifest file format
--------------------
A split manifest directory may contain:
- train.txt
- val.txt
- test.txt

Each file contains one sample ID per line.
Blank lines and lines starting with "#" are ignored.

Usage examples
--------------
In-memory dataset:

    base_dataset = SplitPhyloDataset(data_list=graphs, labels=labels)
    split = DatasetSplit.from_ratios(
        sample_ids=base_dataset.sample_ids,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42,
    )

    train_dataset = base_dataset.subset("train", split)
    val_dataset = base_dataset.subset("val", split)
    test_dataset = base_dataset.subset("test", split)

Disk-backed dataset:

    dataset = SplitPhyloDiskDataset(
        graph_dir="dataset/graphs",
        label_dir="dataset/labels",
    )

    split = DatasetSplit.from_manifest_dir("dataset/splits")
    train_dataset = dataset.subset("train", split)
    val_dataset = dataset.subset("val", split)
    test_dataset = dataset.subset("test", split)

Engineering notes
-----------------
- All indexing is deterministic
- Subsets are lightweight views over a base dataset
- Graphs are cloned before labels/transforms are attached
- Optional caching is supported for disk-backed loading
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)
import numbers
import random
import re

import torch
from torch import Tensor
from torch_geometric.data import Data, Dataset


# ---------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------
PathLike = Union[str, Path]
LabelValue = Union[Tensor, numbers.Real]
LoadedLabelObject = Union[Tensor, Dict[str, Tensor]]
InMemoryLabels = Union[Tensor, Mapping[str, Tensor]]


# ---------------------------------------------------------------------
# Low-level helper functions
# ---------------------------------------------------------------------
def _ensure_data_instance(obj: object, *, source: str) -> Data:
    """
    Validate that an object is a PyG `Data` instance.

    Parameters
    ----------
    obj : object
        Loaded object to validate.

    source : str
        Description used in error messages.

    Returns
    -------
    Data
        Validated `Data` instance.

    Raises
    ------
    TypeError
        If `obj` is not a `torch_geometric.data.Data` object.
    """
    if not isinstance(obj, Data):
        raise TypeError(
            f"Expected a torch_geometric.data.Data object from {source}, "
            f"but got {type(obj).__name__}."
        )
    return obj


def _clone_data(data: Data) -> Data:
    """
    Clone a graph object before mutation.

    This prevents side effects when attaching labels or applying transforms.
    """
    return data.clone()


def _to_tensor(value: LabelValue, *, scalar_dtype: torch.dtype = torch.float32) -> Tensor:
    """
    Normalize a scalar or tensor into a tensor.

    Parameters
    ----------
    value : Tensor or numeric scalar
        Input value.

    scalar_dtype : torch.dtype, default=torch.float32
        Dtype used when converting scalar numerics.

    Returns
    -------
    Tensor
        Tensor value.

    Raises
    ------
    TypeError
        If `value` is neither a tensor nor a numeric scalar.
    """
    if isinstance(value, Tensor):
        return value
    if isinstance(value, numbers.Real):
        return torch.tensor(value, dtype=scalar_dtype)
    raise TypeError(
        f"Expected a torch.Tensor or numeric scalar, got {type(value).__name__}."
    )


def _normalize_label_object(label_obj: object) -> LoadedLabelObject:
    """
    Normalize a loaded label object.

    Accepted label formats
    ----------------------
    - Tensor
    - numeric scalar
    - dict[str, Tensor or numeric scalar]

    Returns
    -------
    LoadedLabelObject
        Normalized label object.

    Raises
    ------
    TypeError
        If the label format is unsupported.
    """
    if isinstance(label_obj, Tensor):
        return label_obj

    if isinstance(label_obj, numbers.Real):
        return torch.tensor(label_obj, dtype=torch.float32)

    if isinstance(label_obj, Mapping):
        normalized: Dict[str, Tensor] = {}
        for key, value in label_obj.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"Task name must be str, got {type(key).__name__}."
                )
            normalized[key] = _to_tensor(value)
        return normalized

    raise TypeError(
        "Unsupported label format. Expected one of: "
        "torch.Tensor, numeric scalar, or dict[str, Tensor|numeric scalar]."
    )


def _sanitize_task_name(task_name: str) -> str:
    """
    Convert an arbitrary task name into a safe attribute suffix.

    Example
    -------
    'speciation-rate' -> 'speciation_rate'
    """
    sanitized = re.sub(r"\W+", "_", task_name).strip("_")
    if not sanitized:
        raise ValueError(f"Invalid task name: {task_name!r}")
    return sanitized


def _attach_label_to_data(data: Data, label_obj: LoadedLabelObject) -> Data:
    """
    Attach a normalized label object to a graph.

    Specification
    -------------
    Single-task:
    - `data.y = tensor`

    Multi-task:
    - `data.y_<task_name> = tensor`
    - `data.task_names = List[str]`
    """
    if isinstance(label_obj, Tensor):
        data.y = label_obj
        return data

    task_names: List[str] = []
    for task_name, task_tensor in label_obj.items():
        safe_name = _sanitize_task_name(task_name)
        setattr(data, f"y_{safe_name}", task_tensor)
        task_names.append(task_name)

    data.task_names = task_names
    return data


def _sorted_pt_files(root: Path, recursive: bool = False) -> List[Path]:
    """
    Collect `.pt` files in deterministic sorted order.

    Parameters
    ----------
    root : Path
        Root directory to scan.

    recursive : bool, default=False
        Whether to scan recursively.

    Returns
    -------
    List[Path]
        Sorted list of `.pt` file paths.
    """
    pattern = "**/*.pt" if recursive else "*.pt"
    return sorted(p for p in root.glob(pattern) if p.is_file())


def _read_sample_id_file(path: Path) -> List[str]:
    """
    Read a text file containing one sample ID per line.

    Parsing rules
    -------------
    - blank lines are ignored
    - lines beginning with '#' are ignored
    - surrounding whitespace is stripped
    """
    sample_ids: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            item = line.strip()
            if not item:
                continue
            if item.startswith("#"):
                continue
            sample_ids.append(item)
    return sample_ids


# ---------------------------------------------------------------------
# Split container
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class DatasetSplit:
    """
    Immutable split definition for sample IDs.

    Attributes
    ----------
    splits : Dict[str, List[str]]
        Mapping from split name to ordered list of sample IDs.

    Specification
    -------------
    - sample IDs must be unique across all splits
    - order inside each split is preserved
    - split names are arbitrary strings, but common names are:
        * train
        * val
        * test

    Example
    -------
    >>> split = DatasetSplit.from_dict({
    ...     "train": ["a", "b", "c"],
    ...     "val": ["d"],
    ...     "test": ["e", "f"],
    ... })
    """

    splits: Dict[str, List[str]]

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """
        Validate structural correctness and uniqueness across splits.
        """
        if not self.splits:
            raise ValueError("DatasetSplit must contain at least one split.")

        seen: Dict[str, str] = {}
        for split_name, sample_ids in self.splits.items():
            if not isinstance(split_name, str):
                raise TypeError(
                    f"Split name must be str, got {type(split_name).__name__}."
                )
            if not isinstance(sample_ids, list):
                raise TypeError(
                    f"Split '{split_name}' must be a list of sample IDs."
                )

            for sample_id in sample_ids:
                if not isinstance(sample_id, str):
                    raise TypeError(
                        f"Sample ID in split '{split_name}' must be str, "
                        f"got {type(sample_id).__name__}."
                    )
                if sample_id in seen:
                    raise ValueError(
                        f"Sample ID '{sample_id}' appears in multiple splits: "
                        f"'{seen[sample_id]}' and '{split_name}'."
                    )
                seen[sample_id] = split_name

    @classmethod
    def from_dict(cls, split_dict: Mapping[str, Sequence[str]]) -> "DatasetSplit":
        """
        Build a split from an explicit mapping.

        Parameters
        ----------
        split_dict : Mapping[str, Sequence[str]]
            Explicit mapping from split name to sample IDs.

        Returns
        -------
        DatasetSplit
        """
        normalized = {k: list(v) for k, v in split_dict.items()}
        return cls(splits=normalized)

    @classmethod
    def from_ratios(
        cls,
        sample_ids: Sequence[str],
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        seed: int = 42,
        shuffle: bool = True,
        split_names: Tuple[str, str, str] = ("train", "val", "test"),
    ) -> "DatasetSplit":
        """
        Create a deterministic random split from sample IDs.

        Parameters
        ----------
        sample_ids : Sequence[str]
            Ordered sample IDs to split.

        train_ratio : float
            Fraction assigned to the training split.

        val_ratio : float
            Fraction assigned to the validation split.

        test_ratio : float
            Fraction assigned to the test split.

        seed : int, default=42
            Random seed used when `shuffle=True`.

        shuffle : bool, default=True
            Whether to shuffle sample IDs before splitting.

        split_names : Tuple[str, str, str], default=("train", "val", "test")
            Names to use for the three resulting splits.

        Returns
        -------
        DatasetSplit

        Notes
        -----
        Ratios must sum to 1.0 within a small numerical tolerance.
        The implementation uses floor-based allocation for the first two splits,
        and assigns the remainder to the last split.
        """
        if len(split_names) != 3:
            raise ValueError("split_names must contain exactly three names.")

        if not sample_ids:
            raise ValueError("sample_ids must not be empty.")

        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 1e-8:
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total:.12f}."
            )

        for name, ratio in zip(split_names, (train_ratio, val_ratio, test_ratio)):
            if ratio < 0:
                raise ValueError(f"Split ratio for '{name}' must be non-negative.")

        ids = list(sample_ids)
        if len(set(ids)) != len(ids):
            raise ValueError("sample_ids must be unique.")

        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(ids)

        n = len(ids)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val

        train_name, val_name, test_name = split_names
        return cls.from_dict(
            {
                train_name: ids[:n_train],
                val_name: ids[n_train:n_train + n_val],
                test_name: ids[n_train + n_val:n_train + n_val + n_test],
            }
        )

    @classmethod
    def from_manifest_dir(
        cls,
        manifest_dir: PathLike,
        split_files: Mapping[str, str] = None,
    ) -> "DatasetSplit":
        """
        Load split definitions from a directory of text manifests.

        Parameters
        ----------
        manifest_dir : str or Path
            Directory containing text files for split definitions.

        split_files : Optional[Mapping[str, str]]
            Mapping from split name to filename.
            Default:
                {
                    "train": "train.txt",
                    "val": "val.txt",
                    "test": "test.txt",
                }

        Returns
        -------
        DatasetSplit
        """
        manifest_dir = Path(manifest_dir)
        if split_files is None:
            split_files = {
                "train": "train.txt",
                "val": "val.txt",
                "test": "test.txt",
            }

        if not manifest_dir.exists():
            raise FileNotFoundError(f"Manifest directory does not exist: {manifest_dir}")
        if not manifest_dir.is_dir():
            raise NotADirectoryError(f"Manifest directory is not a directory: {manifest_dir}")

        split_dict: Dict[str, List[str]] = {}
        for split_name, filename in split_files.items():
            path = manifest_dir / filename
            if not path.exists():
                raise FileNotFoundError(
                    f"Split manifest for '{split_name}' does not exist: {path}"
                )
            split_dict[split_name] = _read_sample_id_file(path)

        return cls.from_dict(split_dict)

    def split_names(self) -> List[str]:
        """
        Return the available split names in insertion order.
        """
        return list(self.splits.keys())

    def sample_ids(self, split_name: str) -> List[str]:
        """
        Return the ordered sample IDs for one split.
        """
        if split_name not in self.splits:
            raise KeyError(f"Unknown split '{split_name}'.")
        return list(self.splits[split_name])

    def contains(self, sample_id: str) -> bool:
        """
        Return whether a sample ID is contained in any split.
        """
        return any(sample_id in ids for ids in self.splits.values())

    def __len__(self) -> int:
        """
        Return the total number of assigned sample IDs across all splits.
        """
        return sum(len(v) for v in self.splits.values())


# ---------------------------------------------------------------------
# Base split-aware dataset subset view
# ---------------------------------------------------------------------
class SplitDatasetView(Dataset):
    """
    Lightweight split-specific view over a base dataset.

    This class does not own graph storage. Instead, it references a base dataset
    and a list of indices corresponding to one split.

    Parameters
    ----------
    base_dataset : _BaseSplitAwareDataset
        The base dataset.

    split_name : str
        Name of the selected split, e.g. "train", "val", or "test".

    indices : Sequence[int]
        Base-dataset indices belonging to the selected split.

    transform : Optional[Callable], default=None
        Optional override transform for the view. If None, the base dataset
        transform behavior is used.

    Specification
    -------------
    - `len(view)` equals the number of indices in the split
    - `view.get(i)` delegates to the corresponding sample in the base dataset
    - returned data is whatever the base dataset returns
    """

    def __init__(
        self,
        base_dataset: "_BaseSplitAwareDataset",
        split_name: str,
        indices: Sequence[int],
        transform: Optional[Callable] = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.split_name = split_name
        self.indices = list(indices)
        self._override_transform = transform
        super().__init__(root=None, transform=transform, pre_transform=None)

    def len(self) -> int:
        """Return number of samples in this split view."""
        return len(self.indices)

    def get(self, idx: int) -> Data:
        """
        Retrieve one sample from the split view.

        Notes
        -----
        If a view-specific transform is provided, it is applied after the base
        dataset retrieval. For this reason, the base dataset should ideally be
        initialized with `transform=None` if split-specific transforms are used.
        """
        data = self.base_dataset[self.indices[idx]]
        if self._override_transform is not None:
            data = self._override_transform(data)
        return data

    @property
    def sample_ids(self) -> List[str]:
        """
        Return the ordered sample IDs in this split.
        """
        return [self.base_dataset.sample_ids[i] for i in self.indices]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"split_name='{self.split_name}', "
            f"num_samples={len(self)}"
            f")"
        )


# ---------------------------------------------------------------------
# Base mixin for split-aware datasets
# ---------------------------------------------------------------------
class _BaseSplitAwareDataset(Dataset):
    """
    Internal base class for split-aware datasets.

    This class defines the common subset construction logic shared by in-memory
    and disk-backed implementations.
    """

    sample_ids: List[str]

    def subset(
        self,
        split_name: str,
        split: DatasetSplit,
        *,
        strict: bool = True,
        transform: Optional[Callable] = None,
    ) -> SplitDatasetView:
        """
        Create a split-specific dataset view.

        Parameters
        ----------
        split_name : str
            Name of the split to select.

        split : DatasetSplit
            Split definition object.

        strict : bool, default=True
            If True, raise an error when the split references sample IDs that do
            not exist in the dataset.
            If False, silently ignore missing sample IDs.

        transform : Optional[Callable], default=None
            Optional transform applied by the view after retrieving samples from
            the base dataset.

        Returns
        -------
        SplitDatasetView
            Split-specific dataset view.
        """
        split_ids = split.sample_ids(split_name)
        id_to_index = {sample_id: idx for idx, sample_id in enumerate(self.sample_ids)}

        missing = [sample_id for sample_id in split_ids if sample_id not in id_to_index]
        if missing and strict:
            preview = "\n".join(missing[:10])
            extra = "" if len(missing) <= 10 else f"\n... and {len(missing) - 10} more"
            raise KeyError(
                f"Split '{split_name}' contains sample IDs not present in the dataset:\n"
                f"{preview}{extra}"
            )

        indices = [id_to_index[sample_id] for sample_id in split_ids if sample_id in id_to_index]
        return SplitDatasetView(
            base_dataset=self,
            split_name=split_name,
            indices=indices,
            transform=transform,
        )

    def build_subsets(
        self,
        split: DatasetSplit,
        *,
        strict: bool = True,
        transform_map: Optional[Mapping[str, Callable]] = None,
    ) -> Dict[str, SplitDatasetView]:
        """
        Build split views for all splits contained in a `DatasetSplit`.

        Parameters
        ----------
        split : DatasetSplit
            Split definition.

        strict : bool, default=True
            Whether missing sample IDs should raise an error.

        transform_map : Optional[Mapping[str, Callable]], default=None
            Optional mapping from split name to a transform used only for that
            split view.

        Returns
        -------
        Dict[str, SplitDatasetView]
            Mapping from split name to split dataset view.
        """
        subsets: Dict[str, SplitDatasetView] = {}
        for split_name in split.split_names():
            split_transform = None if transform_map is None else transform_map.get(split_name)
            subsets[split_name] = self.subset(
                split_name=split_name,
                split=split,
                strict=strict,
                transform=split_transform,
            )
        return subsets


# ---------------------------------------------------------------------
# In-memory split-aware dataset
# ---------------------------------------------------------------------
class SplitPhyloDataset(_BaseSplitAwareDataset):
    """
    In-memory split-aware dataset for phylogenetic graph data.

    Parameters
    ----------
    data_list : Sequence[Data]
        Graphs already loaded in memory.

    labels : Optional[Tensor or Dict[str, Tensor]], default=None
        Per-sample targets.

        Supported forms:
        - single-task:
            Tensor of shape [num_samples, ...]
        - multi-task:
            dict[str, Tensor], where each tensor has first dimension
            equal to `num_samples`

    sample_ids : Optional[Sequence[str]], default=None
        Stable sample identifiers.

        If None, default sample IDs are generated as:
            ["0", "1", ..., "N-1"]

    transform : Optional[Callable], default=None
        Transform applied during sample retrieval.

    pre_transform : Optional[Callable], default=None
        One-time transform applied during initialization.

    Specification
    -------------
    - `len(dataset) == len(data_list)`
    - `dataset.get(i)` returns a cloned graph
    - labels are attached to the cloned graph only
    - original stored graphs are not mutated by retrieval
    - sample IDs are unique and stable
    """

    def __init__(
        self,
        data_list: Sequence[Data],
        labels: Optional[InMemoryLabels] = None,
        sample_ids: Optional[Sequence[str]] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ) -> None:
        self._data_list: List[Data] = [self._validate_graph(data, idx) for idx, data in enumerate(data_list)]
        self._labels = labels
        self._task_names: Optional[List[str]] = None

        self.sample_ids = (
            [str(i) for i in range(len(self._data_list))]
            if sample_ids is None
            else list(sample_ids)
        )

        self._validate_sample_ids()
        self._validate_labels()

        super().__init__(root=None, transform=transform, pre_transform=pre_transform)

        if self.pre_transform is not None:
            self._data_list = [self.pre_transform(_clone_data(data)) for data in self._data_list]

    @staticmethod
    def _validate_graph(data: object, idx: int) -> Data:
        return _ensure_data_instance(data, source=f"data_list[{idx}]")

    def _validate_sample_ids(self) -> None:
        """
        Validate sample ID length, type, and uniqueness.
        """
        if len(self.sample_ids) != len(self._data_list):
            raise ValueError(
                f"Number of sample_ids ({len(self.sample_ids)}) must match "
                f"number of graphs ({len(self._data_list)})."
            )

        if any(not isinstance(sample_id, str) for sample_id in self.sample_ids):
            bad_type = next(type(s).__name__ for s in self.sample_ids if not isinstance(s, str))
            raise TypeError(f"All sample_ids must be strings, found {bad_type}.")

        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("sample_ids must be unique.")

    def _validate_labels(self) -> None:
        """
        Validate label structure against dataset size.
        """
        if self._labels is None:
            return

        n = len(self._data_list)

        if isinstance(self._labels, Tensor):
            if len(self._labels) != n:
                raise ValueError(
                    f"Single-task labels contain {len(self._labels)} samples, "
                    f"but dataset contains {n} graphs."
                )
            return

        if isinstance(self._labels, Mapping):
            self._task_names = list(self._labels.keys())
            for task_name, task_tensor in self._labels.items():
                if not isinstance(task_name, str):
                    raise TypeError(
                        f"Task name must be str, got {type(task_name).__name__}."
                    )
                if not isinstance(task_tensor, Tensor):
                    raise TypeError(
                        f"Task '{task_name}' labels must be a torch.Tensor, "
                        f"got {type(task_tensor).__name__}."
                    )
                if len(task_tensor) != n:
                    raise ValueError(
                        f"Task '{task_name}' contains {len(task_tensor)} samples, "
                        f"but dataset contains {n} graphs."
                    )
            return

        raise TypeError(
            "labels must be one of: None, torch.Tensor, or dict[str, torch.Tensor]."
        )

    def len(self) -> int:
        """Return number of graphs."""
        return len(self._data_list)

    def get(self, idx: int) -> Data:
        """
        Retrieve one sample by index.
        """
        data = _clone_data(self._data_list[idx])

        if self._labels is not None:
            if isinstance(self._labels, Tensor):
                data.y = self._labels[idx]
            else:
                for task_name, task_tensor in self._labels.items():
                    safe_name = _sanitize_task_name(task_name)
                    setattr(data, f"y_{safe_name}", task_tensor[idx])
                data.task_names = list(self._labels.keys())

        data.sample_id = self.sample_ids[idx]

        if self.transform is not None:
            data = self.transform(data)

        return data

    def get_task_names(self) -> Optional[List[str]]:
        """
        Return multi-task names if available.
        """
        return None if self._task_names is None else self._task_names.copy()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"num_graphs={len(self)}, "
            f"has_labels={self._labels is not None}"
            f")"
        )


# ---------------------------------------------------------------------
# Disk-backed split-aware dataset
# ---------------------------------------------------------------------
class SplitPhyloDiskDataset(_BaseSplitAwareDataset):
    """
    Disk-backed split-aware dataset for phylogenetic graph data.

    Parameters
    ----------
    graph_dir : str or Path
        Directory containing graph `.pt` files.

    label_dir : Optional[str or Path], default=None
        Directory containing label `.pt` files.
        Labels are matched to graphs using mirrored relative paths.

    transform : Optional[Callable], default=None
        Transform applied during retrieval.

    pre_filter : Optional[Callable[[Path], bool]], default=None
        Optional file-level filter applied to graph paths before indexing.

    recursive : bool, default=False
        Whether to scan `graph_dir` recursively.

    load_on_cpu : bool, default=True
        If True, use `map_location="cpu"` when calling `torch.load`.

    cache_graphs : bool, default=False
        If True, cache loaded graph objects after first access.

    cache_labels : bool, default=False
        If True, cache loaded labels after first access.

    strict_label_check : bool, default=True
        If True and `label_dir` is set, initialization fails when any graph is
        missing a mirrored label file.

    sample_id_from_relative_path : bool, default=True
        If True, sample IDs are generated from graph relative paths without the
        `.pt` suffix. This is the recommended production behavior.

    Specification
    -------------
    - Graph files are indexed in deterministic sorted order
    - Each graph file must contain a `torch_geometric.data.Data`
    - Label files may contain:
        * Tensor
        * numeric scalar
        * dict[str, Tensor | numeric scalar]
    - Retrieved graphs are cloned before mutation
    - `sample_id` is attached to each returned graph

    Sample ID examples
    ------------------
    graph_dir/file.pt            -> "file"
    graph_dir/sub/file.pt        -> "sub/file"

    Typical usage
    -------------
    >>> dataset = SplitPhyloDiskDataset(
    ...     graph_dir="dataset/graphs",
    ...     label_dir="dataset/labels",
    ...     recursive=True,
    ... )
    >>> split = DatasetSplit.from_manifest_dir("dataset/splits")
    >>> train_dataset = dataset.subset("train", split)
    """

    def __init__(
        self,
        graph_dir: PathLike,
        label_dir: Optional[PathLike] = None,
        transform: Optional[Callable] = None,
        pre_filter: Optional[Callable[[Path], bool]] = None,
        recursive: bool = False,
        load_on_cpu: bool = True,
        cache_graphs: bool = False,
        cache_labels: bool = False,
        strict_label_check: bool = True,
        sample_id_from_relative_path: bool = True,
    ) -> None:
        self.graph_dir = Path(graph_dir)
        self.label_dir = Path(label_dir) if label_dir is not None else None
        self.pre_filter = pre_filter
        self.recursive = recursive
        self.load_on_cpu = load_on_cpu
        self.cache_graphs = cache_graphs
        self.cache_labels = cache_labels
        self.strict_label_check = strict_label_check
        self.sample_id_from_relative_path = sample_id_from_relative_path

        self._graph_paths: List[Path] = []
        self._label_paths: Optional[List[Optional[Path]]] = None
        self.sample_ids: List[str] = []
        self._task_names: Optional[List[str]] = None

        self._graph_cache: Dict[int, Data] = {}
        self._label_cache: Dict[int, LoadedLabelObject] = {}

        self._build_index()

        super().__init__(root=None, transform=transform, pre_transform=None)

    # ---------------------------------------------------------
    # Indexing
    # ---------------------------------------------------------
    def _build_index(self) -> None:
        """
        Build the deterministic file index for the dataset.
        """
        self._validate_directories()

        graph_paths = _sorted_pt_files(self.graph_dir, recursive=self.recursive)
        if self.pre_filter is not None:
            graph_paths = [p for p in graph_paths if self.pre_filter(p)]

        if not graph_paths:
            raise ValueError(f"No .pt graph files found under: {self.graph_dir}")

        self._graph_paths = graph_paths
        self.sample_ids = [self._make_sample_id(p) for p in self._graph_paths]

        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError(
                "Generated sample IDs are not unique. "
                "Check your file structure or sample ID generation logic."
            )

        if self.label_dir is not None:
            self._label_paths = [self._resolve_label_path(p) for p in self._graph_paths]

            if self.strict_label_check:
                missing_pairs = [
                    str(graph_path)
                    for graph_path, label_path in zip(self._graph_paths, self._label_paths)
                    if label_path is None
                ]
                if missing_pairs:
                    preview = "\n".join(missing_pairs[:10])
                    extra = "" if len(missing_pairs) <= 10 else f"\n... and {len(missing_pairs) - 10} more"
                    raise FileNotFoundError(
                        "Missing label files for the following graph files:\n"
                        f"{preview}{extra}"
                    )

            self._infer_task_names_if_possible()

    def _validate_directories(self) -> None:
        """
        Validate graph/label directories.
        """
        if not self.graph_dir.exists():
            raise FileNotFoundError(f"graph_dir does not exist: {self.graph_dir}")
        if not self.graph_dir.is_dir():
            raise NotADirectoryError(f"graph_dir is not a directory: {self.graph_dir}")

        if self.label_dir is not None:
            if not self.label_dir.exists():
                raise FileNotFoundError(f"label_dir does not exist: {self.label_dir}")
            if not self.label_dir.is_dir():
                raise NotADirectoryError(f"label_dir is not a directory: {self.label_dir}")

    def _make_sample_id(self, graph_path: Path) -> str:
        """
        Create the stable sample ID for a graph path.

        Default behavior:
        - relative path to `graph_dir`
        - remove `.pt` suffix
        - use POSIX-style separators for cross-platform stability
        """
        if self.sample_id_from_relative_path:
            relative = graph_path.relative_to(self.graph_dir)
            return relative.with_suffix("").as_posix()

        return graph_path.stem

    def _resolve_label_path(self, graph_path: Path) -> Optional[Path]:
        """
        Resolve the expected label file path for one graph.

        Matching rule
        -------------
        Mirrored relative path:
        - graph_dir/sub/a.pt -> label_dir/sub/a.pt
        """
        assert self.label_dir is not None
        relative = graph_path.relative_to(self.graph_dir)
        candidate = self.label_dir / relative
        return candidate if candidate.exists() and candidate.is_file() else None

    def _infer_task_names_if_possible(self) -> None:
        """
        Best-effort inference of multi-task names from the first available label.
        """
        if self._label_paths is None:
            return

        first_label_path = next((p for p in self._label_paths if p is not None), None)
        if first_label_path is None:
            return

        label_obj = self._load_label_file(first_label_path)
        if isinstance(label_obj, dict):
            self._task_names = list(label_obj.keys())

    # ---------------------------------------------------------
    # Loading helpers
    # ---------------------------------------------------------
    @property
    def _map_location(self):
        return "cpu" if self.load_on_cpu else None

    def _load_graph_file(self, path: Path) -> Data:
        """
        Load and validate a graph file.
        """
        obj = torch.load(path, map_location=self._map_location)
        return _ensure_data_instance(obj, source=str(path))

    def _load_label_file(self, path: Path) -> LoadedLabelObject:
        """
        Load and normalize a label file.
        """
        obj = torch.load(path, map_location=self._map_location)
        return _normalize_label_object(obj)

    def _get_graph(self, idx: int) -> Data:
        """
        Load a graph, optionally using cache.
        """
        if self.cache_graphs and idx in self._graph_cache:
            return self._graph_cache[idx]

        graph = self._load_graph_file(self._graph_paths[idx])
        if self.cache_graphs:
            self._graph_cache[idx] = graph
        return graph

    def _get_label(self, idx: int) -> Optional[LoadedLabelObject]:
        """
        Load a label, optionally using cache.
        """
        if self._label_paths is None:
            return None

        label_path = self._label_paths[idx]
        if label_path is None:
            return None

        if self.cache_labels and idx in self._label_cache:
            return self._label_cache[idx]

        label_obj = self._load_label_file(label_path)
        if self.cache_labels:
            self._label_cache[idx] = label_obj
        return label_obj

    # ---------------------------------------------------------
    # PyG Dataset API
    # ---------------------------------------------------------
    def len(self) -> int:
        """Return the number of indexed samples."""
        return len(self._graph_paths)

    def get(self, idx: int) -> Data:
        """
        Retrieve one dataset sample.

        Returns
        -------
        Data
            Graph with attached labels if available and `sample_id` metadata.
        """
        data = _clone_data(self._get_graph(idx))

        label_obj = self._get_label(idx)
        if label_obj is not None:
            data = _attach_label_to_data(data, label_obj)

        data.sample_id = self.sample_ids[idx]

        if self.transform is not None:
            data = self.transform(data)

        return data

    # ---------------------------------------------------------
    # Convenience methods
    # ---------------------------------------------------------
    def get_graph_path(self, idx: int) -> Path:
        """
        Return the source graph file path for a sample index.
        """
        return self._graph_paths[idx]

    def get_label_path(self, idx: int) -> Optional[Path]:
        """
        Return the source label file path for a sample index, if available.
        """
        if self._label_paths is None:
            return None
        return self._label_paths[idx]

    def get_task_names(self) -> Optional[List[str]]:
        """
        Return inferred multi-task names if available.
        """
        return None if self._task_names is None else self._task_names.copy()

    def export_split_manifests(
        self,
        split: DatasetSplit,
        output_dir: PathLike,
        filenames: Optional[Mapping[str, str]] = None,
        strict: bool = True,
    ) -> None:
        """
        Export split definitions to text manifest files.

        Parameters
        ----------
        split : DatasetSplit
            Split definition to export.

        output_dir : str or Path
            Target directory.

        filenames : Optional[Mapping[str, str]]
            Optional mapping from split name to output filename.
            If not provided, defaults to `<split_name>.txt`.

        strict : bool, default=True
            If True, raise an error when a split references sample IDs that do
            not exist in the current dataset.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        known_ids = set(self.sample_ids)

        for split_name in split.split_names():
            sample_ids = split.sample_ids(split_name)

            if strict:
                missing = [sid for sid in sample_ids if sid not in known_ids]
                if missing:
                    preview = "\n".join(missing[:10])
                    extra = "" if len(missing) <= 10 else f"\n... and {len(missing) - 10} more"
                    raise KeyError(
                        f"Split '{split_name}' contains unknown sample IDs:\n"
                        f"{preview}{extra}"
                    )

            filename = (
                filenames[split_name]
                if filenames is not None and split_name in filenames
                else f"{split_name}.txt"
            )
            path = output_dir / filename

            with path.open("w", encoding="utf-8") as f:
                for sample_id in sample_ids:
                    if not strict and sample_id not in known_ids:
                        continue
                    f.write(f"{sample_id}\n")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"num_graphs={len(self)}, "
            f"graph_dir='{self.graph_dir}', "
            f"label_dir='{self.label_dir}', "
            f"recursive={self.recursive}"
            f")"
        )


# ---------------------------------------------------------------------
# Optional high-level utility functions
# ---------------------------------------------------------------------
def build_ratio_subsets(
    dataset: _BaseSplitAwareDataset,
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
    shuffle: bool = True,
    transform_map: Optional[Mapping[str, Callable]] = None,
) -> Tuple[DatasetSplit, Dict[str, SplitDatasetView]]:
    """
    Convenience helper to build a ratio-based split and immediately create
    dataset subset views.

    Parameters
    ----------
    dataset : _BaseSplitAwareDataset
        Base dataset.

    train_ratio : float
        Train split ratio.

    val_ratio : float
        Validation split ratio.

    test_ratio : float
        Test split ratio.

    seed : int, default=42
        Random seed for deterministic shuffling.

    shuffle : bool, default=True
        Whether to shuffle sample IDs before splitting.

    transform_map : Optional[Mapping[str, Callable]], default=None
        Optional per-split transform mapping.

    Returns
    -------
    Tuple[DatasetSplit, Dict[str, SplitDatasetView]]
        The split definition and the constructed subset views.
    """
    split = DatasetSplit.from_ratios(
        sample_ids=dataset.sample_ids,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        shuffle=shuffle,
    )
    subsets = dataset.build_subsets(split, transform_map=transform_map)
    return split, subsets


def build_manifest_subsets(
    dataset: _BaseSplitAwareDataset,
    manifest_dir: PathLike,
    *,
    split_files: Optional[Mapping[str, str]] = None,
    strict: bool = True,
    transform_map: Optional[Mapping[str, Callable]] = None,
) -> Tuple[DatasetSplit, Dict[str, SplitDatasetView]]:
    """
    Convenience helper to load split manifests and create subset views.

    Parameters
    ----------
    dataset : _BaseSplitAwareDataset
        Base dataset.

    manifest_dir : str or Path
        Directory containing split manifest files.

    split_files : Optional[Mapping[str, str]], default=None
        Optional mapping from split name to filename.

    strict : bool, default=True
        Whether missing sample IDs should raise an error.

    transform_map : Optional[Mapping[str, Callable]], default=None
        Optional per-split transform mapping.

    Returns
    -------
    Tuple[DatasetSplit, Dict[str, SplitDatasetView]]
        The split definition and the constructed subset views.
    """
    split = DatasetSplit.from_manifest_dir(
        manifest_dir=manifest_dir,
        split_files=split_files,
    )
    subsets = dataset.build_subsets(
        split=split,
        strict=strict,
        transform_map=transform_map,
    )
    return split, subsets
