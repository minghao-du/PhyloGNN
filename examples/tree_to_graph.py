"""Self-contained TreeToGraphConverter example."""

import pickle
import sys
import types
from pathlib import Path

import numpy as np
from ete3 import Tree


def _install_torch_compat() -> None:
    """Install a tiny torch/torch_geometric compatibility layer if needed."""
    try:
        import torch  # noqa: F401
        from torch_geometric.data import Data  # noqa: F401
    except ModuleNotFoundError:
        pass
    else:
        return

    class _Tensor:
        def __init__(self, array):
            self._array = np.asarray(array)

        @property
        def dtype(self):
            return self._array.dtype

        @property
        def device(self):
            return "cpu"

        @property
        def shape(self):
            return self._array.shape

        def size(self, dim=None):
            return self._array.shape if dim is None else self._array.shape[dim]

        def dim(self):
            return self._array.ndim

        def numel(self):
            return int(self._array.size)

        def item(self):
            return self._array.item()

        def tolist(self):
            return self._array.tolist()

        def long(self):
            return _Tensor(self._array.astype(np.int64))

        def contiguous(self):
            return self

        def t(self):
            return _Tensor(self._array.T)

        def view(self, *shape):
            return _Tensor(self._array.reshape(shape))

        def expand(self, *shape):
            return _Tensor(np.broadcast_to(self._array, shape))

        def min(self):
            return _Tensor(np.array(self._array.min()))

        def max(self):
            return _Tensor(np.array(self._array.max()))

        def __getitem__(self, item):
            value = self._array[item]
            if np.isscalar(value):
                return value.item()
            return _Tensor(value)

        def __setitem__(self, item, value):
            if isinstance(value, _Tensor):
                value = value._array
            self._array[item] = value

        def __iter__(self):
            for value in self._array:
                yield value

    class _TorchModule(types.ModuleType):
        float32 = np.float32
        long = np.int64
        bool = np.bool_

        @staticmethod
        def tensor(data, dtype=None, device=None):  # noqa: ARG004
            return _Tensor(np.array(data, dtype=dtype))

        @staticmethod
        def zeros(shape, dtype=None, device=None):  # noqa: ARG004
            return _Tensor(np.zeros(shape, dtype=dtype))

        @staticmethod
        def empty(shape, dtype=None, device=None):  # noqa: ARG004
            return _Tensor(np.empty(shape, dtype=dtype))

        @staticmethod
        def full(shape, fill_value, dtype=None, device=None):  # noqa: ARG004
            return _Tensor(np.full(shape, fill_value, dtype=dtype))

        @staticmethod
        def cat(tensors, dim=0):
            arrays = [tensor._array if isinstance(tensor, _Tensor) else np.asarray(tensor) for tensor in tensors]
            return _Tensor(np.concatenate(arrays, axis=dim))

        @staticmethod
        def max(tensor):
            array = tensor._array if isinstance(tensor, _Tensor) else np.asarray(tensor)
            return _Tensor(np.array(array.max()))

        @staticmethod
        def save(obj, path):
            with open(path, "wb") as handle:
                pickle.dump(obj, handle)

        @staticmethod
        def load(path, map_location=None):  # noqa: ARG004
            with open(path, "rb") as handle:
                return pickle.load(handle)

    class _Data:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        @property
        def num_nodes(self):
            return int(self.x.size(0))

    torch_module = _TorchModule("torch")
    torch_module.Tensor = _Tensor
    sys.modules["torch"] = torch_module

    torch_geometric_module = types.ModuleType("torch_geometric")
    torch_geometric_data_module = types.ModuleType("torch_geometric.data")
    torch_geometric_data_module.Data = _Data
    torch_geometric_module.data = torch_geometric_data_module
    sys.modules["torch_geometric"] = torch_geometric_module
    sys.modules["torch_geometric.data"] = torch_geometric_data_module


_install_torch_compat()

# Make the local `src/` package importable when running the script directly.
ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from phylognn.data import TreeFeatureEngineer, TreeToGraphConverter


FEATURE_NAMES = [
    "node_time",
    "time_bin",
    "branch_length",
    "is_tip",
]


def build_demo_tree() -> Tree:
    return Tree("((A:1.0,B:1.5)C:0.5,D:2.0)root:0.0;", format=1)


def main() -> None:
    engineer = TreeFeatureEngineer(num_time_bins=6)
    tree = engineer.add_features(
        build_demo_tree(),
        origin_time=4.0,
        feature_names=FEATURE_NAMES,
        rescale=False,
        inplace=True,
    )

    converter = TreeToGraphConverter(
        feature_names=FEATURE_NAMES,
        add_virtual_nodes=False,
        traversal_strategy=engineer.traversal_strategy,
    )
    data = converter.convert(tree, graph_attrs={"example_name": "tree_to_graph"})

    print("Graph summary")
    print(f"x shape: {tuple(data.x.shape)}")
    print(f"edge_index shape: {tuple(data.edge_index.shape)}")
    print(f"num_nodes: {data.num_nodes}")
    print(f"num_edges: {data.edge_index.size(1)}")
    print(f"feature_names: {converter.output_feature_names}")
    print(f"example_name: {data.example_name}")


if __name__ == "__main__":
    main()
