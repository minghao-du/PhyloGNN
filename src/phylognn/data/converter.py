"""
Tree to graph conversion utilities.

This module provides `TreeToGraphConverter`, which converts an `ete3.Tree`
with precomputed numeric node attributes into a PyTorch Geometric `Data` object.

Overview
--------
The converter assumes that node-level features have already been attached to the
tree, typically by `TreeFeatureEngineer`. It then:

1. Traverses the tree in a stable order
2. Extracts the requested node attributes into a feature matrix
3. Builds graph edges from the tree structure
4. Optionally adds one virtual node per time bin
5. Returns a `torch_geometric.data.Data` object
6. Optionally saves the resulting `Data` object to disk

Typical pipeline
----------------
A common workflow is:

    engineer = TreeFeatureEngineer(...)
    tree = engineer.add_features(tree, origin_time=..., ...)

    converter = TreeToGraphConverter(
        feature_names=engineer.feature_names,
        add_virtual_nodes=True,
        num_time_bins=engineer.num_time_bins,
    )

    data = converter.convert(tree)
    converter.save_data(data, "graph.pt")

Or directly:

    data = converter.convert_and_save(tree, "graph.pt")

Input contract
--------------
Each node in the input tree must already contain every attribute named in
`feature_names`, and each attribute value must be numeric.

Supported numeric types
-----------------------
The converter accepts values that are instances of `numbers.Real`, such as:
- int
- float
- bool
- numpy numeric scalar types compatible with numbers.Real

Graph semantics
---------------
Original tree nodes become graph nodes. Parent-child relationships become graph
edges.

If enabled, virtual nodes are additionally created to represent time bins.
These virtual nodes can be:
- connected to original nodes that share the same `time_bin`
- connected to neighboring virtual nodes in a temporal chain

Produced graph fields
---------------------
The returned `Data` object contains at least:
- x
- edge_index

It may additionally contain:
- edge_type
- original_num_nodes
- virtual_node_mask
- node_type
- node_names
- num_time_bins
- any user-supplied graph-level attributes

Edge type semantics
-------------------
- 0 : tree edge
- 1 : virtual-to-real edge
- 2 : virtual-chain edge

Node type semantics
-------------------
- 0 : original tree node
- 1 : virtual node

Notes on virtual node features
------------------------------
Virtual nodes are initialized with zero features, then certain fields are filled
when available:
- `time_bin` is set to the corresponding bin index
- `extant_sampling_probability` may optionally be copied from original nodes
- `is_virtual_node` is set to 1.0 if appended as an extra feature

Other feature values on virtual nodes remain zero unless explicitly defined by
future extensions.

Save / load support
-------------------
This module also provides helper methods to save and load PyG `Data` objects:

- `save_data(data, path)`
- `load_data(path)`
- `convert_and_save(tree, path, graph_attrs=None)`

These methods are intended to support preprocessing pipelines where graph data
is generated once and reused multiple times.

Scope of responsibility
-----------------------
This class converts node attributes into graph tensors and optionally saves them.
It does not compute biological features itself and does not validate
phylogenetic semantics beyond the structural and numeric requirements described
here.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numbers

import torch
from torch_geometric.data import Data
from ete3 import Tree


PathLike = Union[str, Path]


class TreeToGraphConverter:
    """
    Convert an ETE Tree with node attributes into a PyTorch Geometric graph.

    Parameters
    ----------
    feature_names : Optional[Sequence[str]], default=None
        Ordered list of node attribute names to extract from each tree node.

        Semantics:
        - The order of `feature_names` defines the column order of `data.x`.
        - All nodes must contain every listed attribute.

        If None, the following default order is used:
        - node_time
        - time_bin
        - is_internal
        - is_tip
        - is_fossil
        - is_extant
        - is_sampled_ancestor
        - is_not_sampled_ancestor
        - branch_length
        - extant_sampling_probability

        Recommendation:
        - Use `TreeFeatureEngineer.feature_names` for stable compatibility.

    add_virtual_nodes : bool, default=False
        Whether to add one virtual node per time bin.

        If True:
        - `feature_names` must include `"time_bin"`
        - `num_time_bins` may be provided explicitly or inferred from the tree

    num_time_bins : Optional[int], default=None
        Number of time bins used for virtual nodes.

        Behavior:
        - If `add_virtual_nodes=False`, this argument is ignored.
        - If `add_virtual_nodes=True` and `num_time_bins` is provided:
              it must be at least 2
        - If `add_virtual_nodes=True` and `num_time_bins` is None:
              it is inferred from the maximum observed original-node `time_bin`
              as `max(time_bin) + 1`

    traversal_strategy : str, default="preorder"
        Node traversal strategy used to assign graph node indices.

        Allowed values:
        - "preorder"
        - "postorder"
        - "levelorder"

        This parameter affects node indexing order and any metadata aligned with
        that order, such as `node_names`.

    bidirectional : bool, default=True
        Whether every constructed edge should be added in both directions.

        If True:
        - each tree edge parent->child is accompanied by child->parent
        - the same applies to virtual edges

    connect_virtual_to_real : bool, default=True
        Only relevant when `add_virtual_nodes=True`.

        If True:
        - for each time bin `b`, the virtual node representing bin `b` is
          connected to every original node whose `time_bin == b`

    connect_virtual_chain : bool, default=True
        Only relevant when `add_virtual_nodes=True`.

        If True:
        - virtual node `b` is connected to virtual node `b + 1` for all adjacent
          time bins

    append_is_virtual_feature : bool, default=True
        Whether to append an extra feature column named `is_virtual_node`.

        Semantics:
        - original nodes receive value 0.0
        - virtual nodes receive value 1.0

        This modifies the final output feature dimensionality.

    preserve_node_names : bool, default=True
        Whether to attach `data.node_names`.

        If enabled:
        - original node names are taken from `node.name`
        - unnamed original nodes become empty strings
        - virtual node names are generated as:
              "__virtual_time_bin_i__"

    copy_sampling_prob_to_virtual : bool, default=True
        When virtual nodes are used and `extant_sampling_probability` is present
        in `feature_names`, this flag controls whether that feature is copied
        from the first original node to all virtual nodes.

    Attributes
    ----------
    feature_names : Tuple[str, ...]
        Ordered input feature names extracted from original nodes.

    output_feature_names : List[str]
        Ordered output feature names in `data.x`. This equals `feature_names`
        unless `append_is_virtual_feature=True`, in which case the final column
        is `"is_virtual_node"`.

    Constants
    ---------
    EDGE_TYPE_TREE = 0
    EDGE_TYPE_VIRTUAL_TO_REAL = 1
    EDGE_TYPE_VIRTUAL_CHAIN = 2

    NODE_TYPE_ORIGINAL = 0
    NODE_TYPE_VIRTUAL = 1

    Output contract
    ---------------
    `convert()` returns a `torch_geometric.data.Data` object containing:

    Required fields:
    - x : FloatTensor of shape [num_nodes, num_features]
    - edge_index : LongTensor of shape [2, num_edges]

    Additional standard fields:
    - edge_type : LongTensor of shape [num_edges]
    - original_num_nodes : int
    - virtual_node_mask : BoolTensor of shape [num_nodes]
    - node_type : LongTensor of shape [num_nodes]

    Optional fields:
    - node_names : List[str], if `preserve_node_names=True`
    - num_time_bins : int, if virtual nodes are added
    - arbitrary graph-level attributes passed via `graph_attrs`

    Save/load methods
    -----------------
    - `save_data(data, path)`:
        Save a PyG Data object to disk using `torch.save`.

    - `load_data(path)`:
        Load a previously saved PyG Data object from disk using `torch.load`.

    - `convert_and_save(tree, path, graph_attrs=None)`:
        Convert a tree to `Data`, save it, and return the saved object.

    Examples
    --------
    Basic conversion:

    >>> converter = TreeToGraphConverter(
    ...     feature_names=["node_time", "time_bin", "is_tip"],
    ...     add_virtual_nodes=False,
    ... )
    >>> data = converter.convert(tree)

    With virtual nodes:

    >>> converter = TreeToGraphConverter(
    ...     feature_names=engineer.feature_names,
    ...     add_virtual_nodes=True,
    ...     num_time_bins=engineer.num_time_bins,
    ... )
    >>> data = converter.convert(tree)

    Save to disk:

    >>> converter.save_data(data, "graph.pt")

    Convert and save in one step:

    >>> data = converter.convert_and_save(tree, "graph.pt")
    """

    DEFAULT_FEATURE_NAMES = (
        "node_time",
        "time_bin",
        "is_internal",
        "is_tip",
        "is_fossil",
        "is_extant",
        "is_sampled_ancestor",
        "is_not_sampled_ancestor",
        "branch_length",
        "extant_sampling_probability",
    )

    EDGE_TYPE_TREE = 0
    EDGE_TYPE_VIRTUAL_TO_REAL = 1
    EDGE_TYPE_VIRTUAL_CHAIN = 2

    NODE_TYPE_ORIGINAL = 0
    NODE_TYPE_VIRTUAL = 1

    VALID_TRAVERSALS = {"preorder", "postorder", "levelorder"}
    IS_VIRTUAL_FEATURE_NAME = "is_virtual_node"

    def __init__(
        self,
        feature_names: Optional[Sequence[str]] = None,
        add_virtual_nodes: bool = False,
        num_time_bins: Optional[int] = None,
        traversal_strategy: str = "preorder",
        bidirectional: bool = True,
        connect_virtual_to_real: bool = True,
        connect_virtual_chain: bool = True,
        append_is_virtual_feature: bool = True,
        preserve_node_names: bool = True,
        copy_sampling_prob_to_virtual: bool = True,
    ):
        self.feature_names = (
            tuple(feature_names) if feature_names is not None else self.DEFAULT_FEATURE_NAMES
        )
        self.add_virtual_nodes = add_virtual_nodes
        self.num_time_bins = num_time_bins
        self.traversal_strategy = traversal_strategy
        self.bidirectional = bidirectional
        self.connect_virtual_to_real = connect_virtual_to_real
        self.connect_virtual_chain = connect_virtual_chain
        self.append_is_virtual_feature = append_is_virtual_feature
        self.preserve_node_names = preserve_node_names
        self.copy_sampling_prob_to_virtual = copy_sampling_prob_to_virtual

        self._validate_init_params()

    @property
    def output_feature_names(self) -> Tuple[str, ...]:
        """
        Final ordered feature names in `data.x`.

        Returns
        -------
        Tuple[str, ...]
            If `append_is_virtual_feature=False`, this equals `feature_names`.

            If `append_is_virtual_feature=True`, the final column is:
            - "is_virtual_node"
        """
        if self.append_is_virtual_feature:
            return self.feature_names + (self.IS_VIRTUAL_FEATURE_NAME,)
        return self.feature_names

    def convert(
        self,
        tree: Tree,
        graph_attrs: Optional[Dict[str, object]] = None,
    ) -> Data:
        """
        Convert a single ETE Tree into a PyTorch Geometric `Data` object.

        Parameters
        ----------
        tree : ete3.Tree
            Input tree whose nodes already contain all required features.

        graph_attrs : Optional[Dict[str, object]], default=None
            Optional graph-level attributes to attach to the returned `Data`
            object.

            Example:
                {
                    "tree_id": "tree_001",
                    "origin_time": 10.0
                }

        Returns
        -------
        torch_geometric.data.Data
            Graph representation of the input tree.

        Raises
        ------
        ValueError
            If the tree is empty.

        AttributeError
            If any node is missing a required feature.

        TypeError
            If any feature value is non-numeric.

        ValueError
            If virtual-node construction is requested but required settings are
            invalid.

        Output fields
        -------------
        The returned `Data` includes at least:
        - x
        - edge_index
        - edge_type
        - original_num_nodes
        - virtual_node_mask
        - node_type

        It may additionally include:
        - node_names
        - num_time_bins
        - user-provided graph_attrs
        """
        nodes = list(tree.traverse(self.traversal_strategy))
        if not nodes:
            raise ValueError("Cannot convert an empty tree")

        x, edge_index, edge_type, node_names = self._extract_features_and_edges(nodes)
        num_original_nodes = x.size(0)

        # Optionally append a binary feature that marks whether a node is virtual.
        # At this stage all nodes are original tree nodes, so the appended value is 0.
        if self.append_is_virtual_feature:
            x = torch.cat([x, torch.zeros((num_original_nodes, 1), dtype=x.dtype)], dim=1)

        data = Data(x=x, edge_index=edge_index)
        data.edge_type = edge_type
        data.original_num_nodes = num_original_nodes

        if self.preserve_node_names:
            data.node_names = node_names

        # Attach arbitrary user-defined graph-level metadata.
        if graph_attrs:
            for key, value in graph_attrs.items():
                setattr(data, key, value)

        if self.add_virtual_nodes:
            data = self._add_virtual_nodes(data)

        # Mark virtual nodes explicitly for downstream convenience.
        data.virtual_node_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.virtual_node_mask[data.original_num_nodes :] = True

        # Node type coding:
        # 0 = original tree node
        # 1 = virtual node
        data.node_type = torch.full(
            (data.num_nodes,),
            self.NODE_TYPE_ORIGINAL,
            dtype=torch.long,
        )
        data.node_type[data.original_num_nodes :] = self.NODE_TYPE_VIRTUAL

        return data

    def convert_and_save(
        self,
        tree: Tree,
        path: PathLike,
        graph_attrs: Optional[Dict[str, object]] = None,
        create_dirs: bool = True,
    ) -> Data:
        """
        Convert a tree to a PyG `Data` object and save it to disk.

        This is a convenience wrapper equivalent to:

            data = converter.convert(tree, graph_attrs=graph_attrs)
            converter.save_data(data, path, create_dirs=create_dirs)

        Parameters
        ----------
        tree : ete3.Tree
            Input tree whose nodes already contain all required features.

        path : str or pathlib.Path
            Output file path. Typical extension is `.pt`.

        graph_attrs : Optional[Dict[str, object]], default=None
            Optional graph-level attributes to attach before saving.

        create_dirs : bool, default=True
            If True, automatically create parent directories if they do not exist.

        Returns
        -------
        torch_geometric.data.Data
            The converted and saved `Data` object.
        """
        data = self.convert(tree, graph_attrs=graph_attrs)
        self.save_data(data, path=path, create_dirs=create_dirs)
        return data

    def save_data(
        self,
        data: Data,
        path: PathLike,
        create_dirs: bool = True,
    ) -> None:
        """
        Save a PyTorch Geometric `Data` object to disk.

        Storage format
        --------------
        This method uses `torch.save(data, path)`, which serializes the complete
        `Data` object for later reuse.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The graph object to save.

        path : str or pathlib.Path
            Destination path.

        create_dirs : bool, default=True
            If True, create the parent directory/directories automatically.

        Raises
        ------
        TypeError
            If `data` is not a PyG `Data` object.

        Notes
        -----
        Saved files can be loaded later using `load_data(path)`.
        """
        if not isinstance(data, Data):
            raise TypeError(
                f"data must be a torch_geometric.data.Data instance, got {type(data).__name__}"
            )

        path = Path(path)
        if create_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(data, path)

    @staticmethod
    def load_data(path: PathLike, map_location=None) -> Data:
        """
        Load a previously saved PyTorch Geometric `Data` object from disk.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to a file previously saved by `save_data()` or `convert_and_save()`.

        map_location : optional
            Passed through to `torch.load`. Useful when loading data saved on a
            different device.

        Returns
        -------
        torch_geometric.data.Data
            The loaded graph object.

        Raises
        ------
        TypeError
            If the loaded object is not a PyG `Data` instance.
        """
        path = Path(path)
        data = torch.load(path, map_location=map_location)

        if not isinstance(data, Data):
            raise TypeError(
                f"Loaded object is not a torch_geometric.data.Data instance, "
                f"got {type(data).__name__}"
            )

        return data

    def _extract_features_and_edges(
        self,
        nodes: List[Tree],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        """
        Extract node features and tree edges from an ordered list.

        Parameters
        ----------
        nodes : List[ete3.Tree]
            Ordered nodes produced by tree traversal.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]
            A tuple:
            - x : FloatTensor [num_nodes, num_features]
            - edge_index : LongTensor [2, num_edges]
            - edge_type : LongTensor [num_edges]
            - node_names : List[str]

        Validation performed
        --------------------
        For every node and every requested feature:
        - the attribute must exist
        - the attribute value must be numeric

        Tree edge construction
        ----------------------
        For every parent-child relation:
        - add parent -> child
        - if bidirectional=True, also add child -> parent

        All such edges receive edge type:
        - EDGE_TYPE_TREE = 0
        """
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}

        feature_matrix: List[List[float]] = []
        edge_list: List[List[int]] = []
        edge_types: List[int] = []
        node_names: List[str] = []

        for node in nodes:
            node_names.append(node.name if getattr(node, "name", "") else "")

            row: List[float] = []
            for feature_name in self.feature_names:
                if not hasattr(node, feature_name):
                    raise AttributeError(
                        f"Node '{node.name}' is missing required attribute '{feature_name}'. "
                        f"Did you forget to run TreeFeatureEngineer.add_features()?"
                    )

                value = getattr(node, feature_name)
                if not isinstance(value, numbers.Real):
                    raise TypeError(
                        f"Feature '{feature_name}' on node '{node.name}' must be numeric, "
                        f"got {type(value).__name__}"
                    )

                row.append(float(value))

            feature_matrix.append(row)

            parent_idx = node_to_idx[node]
            for child in node.children:
                child_idx = node_to_idx[child]
                edge_list.append([parent_idx, child_idx])
                edge_types.append(self.EDGE_TYPE_TREE)

                if self.bidirectional:
                    edge_list.append([child_idx, parent_idx])
                    edge_types.append(self.EDGE_TYPE_TREE)

        x = torch.tensor(feature_matrix, dtype=torch.float32)

        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_type = torch.tensor(edge_types, dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_type = torch.empty((0,), dtype=torch.long)

        return x, edge_index, edge_type, node_names

    def _infer_num_time_bins(self, data: Data) -> int:
        """
        Infer the number of time bins from original-node `time_bin` values.

        Rule
        ----
        If original-node time bins are:
            {b1, b2, ..., bk}

        then:
            inferred_num_time_bins = max(b1, ..., bk) + 1

        Returns
        -------
        int
            Inferred number of time bins.

        Raises
        ------
        ValueError
            If `time_bin` is not available in `feature_names`.

        ValueError
            If the original node set is empty.

        ValueError
            If the inferred result is less than 2.
        """
        if "time_bin" not in self.feature_names:
            raise ValueError("Cannot infer num_time_bins without 'time_bin' in feature_names")

        time_bin_idx = self.feature_names.index("time_bin")
        original_time_bins = data.x[: data.original_num_nodes, time_bin_idx]

        if original_time_bins.numel() == 0:
            raise ValueError("Cannot infer num_time_bins from empty original node set")

        inferred = int(torch.max(original_time_bins).item()) + 1
        if inferred < 2:
            raise ValueError(f"Inferred num_time_bins={inferred}, but it must be at least 2")

        return inferred

    def _add_virtual_nodes(self, data: Data) -> Data:
        """
        Add virtual time-bin nodes to an existing graph.

        Preconditions
        -------------
        - `feature_names` must contain "time_bin"
        - `data` must already represent original tree nodes
        - original-node `time_bin` values must be integer-like
        - original-node `time_bin` values must lie in `[0, num_time_bins - 1]`

        Virtual node indexing
        ---------------------
        If the graph currently has `original_num_nodes` original nodes, then the
        virtual node for time bin `b` receives index:

            original_num_nodes + b

        Virtual feature construction
        ----------------------------
        Virtual node feature vectors are initialized to zeros, then:
        - `time_bin` is set to the bin index
        - `extant_sampling_probability` may optionally be copied
        - if appended, `is_virtual_node` is set to 1.0

        Edge construction
        -----------------
        1. Virtual-to-real edges
           For each time bin `b`, connect virtual node `V_b` to every original
           node with `time_bin == b`.

           Edge type:
           - EDGE_TYPE_VIRTUAL_TO_REAL = 1

        2. Virtual chain edges
           For each adjacent pair `(b, b+1)`, connect `V_b` to `V_(b+1)`.

           Edge type:
           - EDGE_TYPE_VIRTUAL_CHAIN = 2

        Returns
        -------
        torch_geometric.data.Data
            The updated graph with virtual nodes and corresponding edges.

        Raises
        ------
        ValueError
            If `"time_bin"` is not present in `feature_names`.

        ValueError
            If any original-node `time_bin` lies outside the configured range
            `[0, num_time_bins - 1]`.

        AssertionError
            If any original-node `time_bin` is not integer-like before
            conversion to `torch.long`.
        """
        if "time_bin" not in self.feature_names:
            raise ValueError("feature_names must include 'time_bin' when add_virtual_nodes=True")

        num_time_bins = (
            self.num_time_bins
            if self.num_time_bins is not None
            else self._infer_num_time_bins(data)
        )

        num_original_nodes = data.original_num_nodes
        total_num_features = data.x.size(1)
        time_bin_idx = self.feature_names.index("time_bin")

        # Initialize virtual node features to zeros.
        virtual_x = torch.zeros(
            (num_time_bins, total_num_features),
            dtype=data.x.dtype,
            device=data.x.device,
        )

        # Fill the time_bin feature for each virtual node.
        for bin_idx in range(num_time_bins):
            virtual_x[bin_idx, time_bin_idx] = float(bin_idx)

        # Optionally copy extant_sampling_probability from the first original node.
        if (
            self.copy_sampling_prob_to_virtual
            and "extant_sampling_probability" in self.feature_names
            and num_original_nodes > 0
        ):
            prob_idx = self.feature_names.index("extant_sampling_probability")
            virtual_x[:, prob_idx] = data.x[0, prob_idx]

        # If enabled, mark virtual nodes in the appended feature column.
        if self.append_is_virtual_feature:
            virtual_x[:, -1] = 1.0

        new_edges: List[List[int]] = []
        new_edge_types: List[int] = []

        original_time_bins = data.x[:num_original_nodes, time_bin_idx]
        rounded_time_bins = torch.round(original_time_bins)
        assert torch.allclose(
            original_time_bins,
            rounded_time_bins,
            atol=1e-6,
            rtol=0.0,
        ), "Original-node time_bin values must be integer-like before conversion."

        original_time_bins = rounded_time_bins.long()
        invalid_mask = (original_time_bins < 0) | (original_time_bins >= num_time_bins)
        if torch.any(invalid_mask):
            invalid_bins = sorted(set(original_time_bins[invalid_mask].tolist()))
            raise ValueError(
                "Original-node time_bin values fall outside configured range "
                f"[0, {num_time_bins - 1}]: {invalid_bins}"
            )

        bin_to_node_indices = {i: [] for i in range(num_time_bins)}

        for node_idx, bin_value in enumerate(original_time_bins.tolist()):
            bin_to_node_indices[bin_value].append(node_idx)

        # Connect each virtual node to original nodes in the same time bin.
        if self.connect_virtual_to_real:
            for bin_idx in range(num_time_bins):
                virtual_idx = num_original_nodes + bin_idx
                for node_idx in bin_to_node_indices[bin_idx]:
                    new_edges.append([virtual_idx, node_idx])
                    new_edge_types.append(self.EDGE_TYPE_VIRTUAL_TO_REAL)

                    if self.bidirectional:
                        new_edges.append([node_idx, virtual_idx])
                        new_edge_types.append(self.EDGE_TYPE_VIRTUAL_TO_REAL)

        # Connect adjacent virtual nodes as a chain through time bins.
        if self.connect_virtual_chain:
            for bin_idx in range(num_time_bins - 1):
                a = num_original_nodes + bin_idx
                b = num_original_nodes + bin_idx + 1
                new_edges.append([a, b])
                new_edge_types.append(self.EDGE_TYPE_VIRTUAL_CHAIN)

                if self.bidirectional:
                    new_edges.append([b, a])
                    new_edge_types.append(self.EDGE_TYPE_VIRTUAL_CHAIN)

        data.x = torch.cat([data.x, virtual_x], dim=0)

        if new_edges:
            new_edge_index = (
                torch.tensor(
                    new_edges,
                    dtype=torch.long,
                    device=data.x.device,
                )
                .t()
                .contiguous()
            )
            new_edge_type = torch.tensor(
                new_edge_types,
                dtype=torch.long,
                device=data.x.device,
            )
            data.edge_index = torch.cat([data.edge_index, new_edge_index], dim=1)
            data.edge_type = torch.cat([data.edge_type, new_edge_type], dim=0)

        if self.preserve_node_names and hasattr(data, "node_names"):
            data.node_names = data.node_names + [
                f"__virtual_time_bin_{i}__" for i in range(num_time_bins)
            ]

        data.num_time_bins = num_time_bins
        return data

    def _validate_init_params(self) -> None:
        """Validate initialization parameters."""
        if not self.feature_names:
            raise ValueError("feature_names must contain at least one feature")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must not contain duplicates")

        if self.traversal_strategy not in self.VALID_TRAVERSALS:
            raise ValueError(
                f"Invalid traversal_strategy='{self.traversal_strategy}'. "
                f"Valid options: {sorted(self.VALID_TRAVERSALS)}"
            )

        if self.add_virtual_nodes:
            if "time_bin" not in self.feature_names:
                raise ValueError(
                    "feature_names must include 'time_bin' when add_virtual_nodes=True"
                )
            if self.num_time_bins is not None and self.num_time_bins < 2:
                raise ValueError(f"num_time_bins must be at least 2, got {self.num_time_bins}")

    def __repr__(self) -> str:
        return (
            "TreeToGraphConverter("
            f"num_features={len(self.feature_names)}, "
            f"output_num_features={len(self.output_feature_names)}, "
            f"add_virtual_nodes={self.add_virtual_nodes}, "
            f"num_time_bins={self.num_time_bins}, "
            f"traversal_strategy='{self.traversal_strategy}', "
            f"bidirectional={self.bidirectional}, "
            f"connect_virtual_to_real={self.connect_virtual_to_real}, "
            f"connect_virtual_chain={self.connect_virtual_chain}, "
            f"append_is_virtual_feature={self.append_is_virtual_feature})"
        )
