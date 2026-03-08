"""
Feature engineering utilities for phylogenetic trees.

This module provides `TreeFeatureEngineer`, a utility for computing and attaching
node-level features to an `ete3.Tree`.

Overview
--------
`TreeFeatureEngineer` operates directly on ETE tree nodes and writes features as
node attributes via `node.add_feature(...)`. These features can later be consumed
by downstream modules such as `TreeToGraphConverter`.

Processing model
----------------
Given:
- an input `ete3.Tree`
- an externally provided `origin_time`

the engineer computes feature values for each node and stores them on the node.

Core definitions
----------------
1. Branch length
   Each node stores the branch length to its parent in `node.dist`.

2. Root
   The root node is obtained by `tree.get_tree_root()`.

3. Node time
   For a node `u`, its time is defined as:

       node_time(u) = origin_time - distance(root, u)

   where `distance(root, u)` is the path-length distance from the root to `u`.

4. Time bin
   Continuous node time is discretized into an integer bin in
   `[0, num_time_bins - 1]`.

Built-in features
-----------------
The following built-in features are registered by default:

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

Feature dependency rules
------------------------
Some features depend on others:

- time_bin depends on node_time
- is_fossil depends on node_time
- is_extant depends on node_time
- is_sampled_ancestor depends on is_fossil
- is_not_sampled_ancestor depends on is_fossil

Dependencies are resolved automatically.

Important notes
---------------
1. Ordered feature names
   `self.feature_names` stores the registered feature names in a stable order.
   This is the preferred attribute to pass to graph converters.

2. Available feature set
   `self.available_features` stores the same names as a set for membership checks.

3. Rescaling
   If requested, branch lengths are rescaled so that the mean of all non-zero
   branch lengths becomes 1. The provided `origin_time` is rescaled by the same
   factor.

4. Floating-point tolerance
   Comparisons to zero and to `origin_time` use `time_tolerance` to avoid
   numerical instability around the boundaries.

Scope of responsibility
-----------------------
This class is responsible only for feature computation and attachment to tree
nodes. It does not build graph objects and does not validate whether the chosen
feature definitions fully match any specific phylogenetic convention.
"""

from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple
import math

from ete3 import Tree


FeatureFunction = Callable[[dict], None]


class TreeFeatureEngineer:
    """
    Add computed node-level features to an ETE Tree.

    Parameters
    ----------
    num_time_bins : int, default=101
        Number of time bins used to discretize continuous node times.

        Constraints:
        - Must be at least 2.

        Semantic meaning:
        - The discrete time bin range is always:
              0, 1, ..., num_time_bins - 1

    extant_sampling_probability : float, default=1.0
        Sampling probability assigned to the feature
        `extant_sampling_probability`.

        Constraints:
        - Must lie in [0, 1].

        Current behavior:
        - The same scalar value is assigned to every node.

    custom_features : Optional[Dict[str, FeatureFunction]], default=None
        Optional custom feature registry.

        Expected format:
            {
                "feature_name": feature_function
            }

        Each custom feature function must accept a `context` dictionary and
        attach the computed value to `context["node"]`, typically using:

            node.add_feature("feature_name", value)

        The context dictionary contains at least:
        - "node": current node
        - "root": root node
        - "origin_time": origin time used for this tree

    traversal_strategy : str, default="preorder"
        Tree traversal strategy used when iterating over nodes.

        Allowed values:
        - "preorder"
        - "postorder"
        - "levelorder"

        This affects traversal order only. It does not change the mathematical
        definition of built-in features.

    time_tolerance : float, default=1e-8
        Numerical tolerance used for floating-point comparisons, especially when
        testing whether:
        - node_time is effectively zero
        - node_time is effectively equal to origin_time
        - branch length is effectively zero

        Constraints:
        - Must be non-negative.

    Attributes
    ----------
    num_time_bins : int
        Number of time bins used for discretization.

    extant_sampling_probability : float
        Stored sampling probability scalar.

    traversal_strategy : str
        Node traversal strategy used by the engineer.

    time_tolerance : float
        Floating-point comparison tolerance.

    feature_names : List[str]
        Ordered list of all registered feature names. This order is stable and
        should be used whenever a downstream consumer needs a deterministic
        feature-column order.

    available_features : Set[str]
        Unordered set of all registered feature names. Useful for membership
        checks and validation.

    Examples
    --------
    Basic use:

    >>> from ete3 import Tree
    >>> engineer = TreeFeatureEngineer(num_time_bins=10)
    >>> tree = Tree("((A:1,B:2)C:3,D:4)E:0;", format=1)
    >>> tree = engineer.add_features(tree, origin_time=10.0, rescale=True)

    Selective features:

    >>> tree = engineer.add_features(
    ...     tree,
    ...     origin_time=10.0,
    ...     feature_names=["node_time", "time_bin", "is_tip"],
    ...     rescale=False,
    ... )

    Use with a graph converter:

    >>> feature_order = engineer.feature_names
    """

    VALID_TRAVERSALS = {"preorder", "postorder", "levelorder"}

    def __init__(
        self,
        num_time_bins: int = 101,
        extant_sampling_probability: float = 1.0,
        custom_features: Optional[Dict[str, FeatureFunction]] = None,
        traversal_strategy: str = "preorder",
        time_tolerance: float = 1e-8,
    ):
        self._validate_parameters(
            num_time_bins=num_time_bins,
            extant_sampling_probability=extant_sampling_probability,
            traversal_strategy=traversal_strategy,
            time_tolerance=time_tolerance,
        )

        self.num_time_bins = num_time_bins
        self.extant_sampling_probability = extant_sampling_probability
        self.traversal_strategy = traversal_strategy
        self.time_tolerance = time_tolerance

        self._feature_registry: "OrderedDict[str, FeatureFunction]" = OrderedDict([
            ("node_time", self._add_node_time),
            ("time_bin", self._add_time_bin),
            ("is_internal", self._add_is_internal),
            ("is_tip", self._add_is_tip),
            ("is_fossil", self._add_is_fossil),
            ("is_extant", self._add_is_extant),
            ("is_sampled_ancestor", self._add_is_sampled_ancestor),
            ("is_not_sampled_ancestor", self._add_is_not_sampled_ancestor),
            ("branch_length", self._add_branch_length),
            ("extant_sampling_probability", self._add_extant_sampling_probability),
        ])

        if custom_features:
            for name, fn in custom_features.items():
                self._feature_registry[name] = fn

        self.feature_names: List[str] = list(self._feature_registry.keys())
        self.available_features: Set[str] = set(self.feature_names)

    def rescale_tree(
        self,
        tree: Tree,
        origin_time: float,
        inplace: bool = True,
    ) -> Tuple[Tree, float, float]:
        """
        Rescale tree branch lengths so that the mean of all non-zero branch
        lengths becomes 1.

        Formal definition
        -----------------
        Let the set of strictly positive branch lengths be:

            L = { node.dist | node in tree, node.dist > 0 }

        Then:

            mean_length = mean(L)
            scale_factor = 1 / mean_length

        Every branch length is multiplied by `scale_factor`, and the provided
        `origin_time` is also multiplied by the same factor.

        Parameters
        ----------
        tree : ete3.Tree
            Input tree to rescale.

        origin_time : float
            Original origin time associated with the tree.

            Constraint:
            - Must be positive.

        inplace : bool, default=True
            If True, modify the input tree directly.
            If False, operate on a copy and return the copy.

        Returns
        -------
        Tuple[Tree, float, float]
            A tuple of:
            - rescaled_tree : Tree
                The rescaled tree
            - scale_factor : float
                The multiplicative factor applied to all branch lengths
            - new_origin_time : float
                The rescaled origin time

        Raises
        ------
        ValueError
            If `origin_time <= 0`.

        ValueError
            If the tree contains no non-zero branch lengths.

        Guarantees
        ----------
        - All node distances are multiplied by the same scalar.
        - Zero-length edges remain zero.
        - If `inplace=False`, the input tree is not modified.
        """
        if origin_time <= 0:
            raise ValueError(f"origin_time must be positive, got {origin_time}")

        if not inplace:
            tree = tree.copy()

        non_zero_lengths = [
            node.dist
            for node in tree.traverse(self.traversal_strategy)
            if node.dist > 0
        ]

        if not non_zero_lengths:
            raise ValueError("Tree has no non-zero branch lengths to rescale")

        mean_length = sum(non_zero_lengths) / len(non_zero_lengths)
        scale_factor = 1.0 / mean_length

        for node in tree.traverse(self.traversal_strategy):
            node.dist *= scale_factor

        new_origin_time = origin_time * scale_factor
        return tree, scale_factor, new_origin_time

    def add_features(
        self,
        tree: Tree,
        origin_time: float,
        feature_names: Optional[Sequence[str]] = None,
        rescale: bool = True,
        inplace: bool = True,
    ) -> Tree:
        """
        Compute and attach requested features to every node in a tree.

        Workflow
        --------
        1. Optionally copy the tree.
        2. Optionally rescale branch lengths and origin_time.
        3. Determine which features to add.
        4. Traverse all nodes and compute the requested features.
        5. Store each feature as a node attribute.

        Parameters
        ----------
        tree : ete3.Tree
            Input tree.

        origin_time : float
            Root age / tree origin time.

            Constraint:
            - Must be positive.

        feature_names : Optional[Sequence[str]], default=None
            Requested features to compute.

            Behavior:
            - If None, all registered features are added.
            - If provided, only those features are explicitly requested.

            Validation:
            - Every requested feature name must exist in `available_features`.

        rescale : bool, default=True
            Whether to rescale the tree before feature computation.

            If True:
            - branch lengths are rescaled by `rescale_tree()`
            - origin_time is rescaled by the same factor
            - feature computation uses the rescaled values

        inplace : bool, default=True
            If True, modify the input tree in place.
            If False, compute features on a copy and return the copy.

        Returns
        -------
        ete3.Tree
            The tree with features attached to its nodes.

        Raises
        ------
        ValueError
            If `origin_time <= 0`.

        ValueError
            If an unknown feature name is requested.

        Notes
        -----
        Dependency features are computed automatically when needed. For example:
        - requesting `time_bin` will also compute `node_time` if missing
        - requesting `is_sampled_ancestor` will also compute `is_fossil` if missing

        However, users should still treat the explicitly requested features as the
        primary contract.
        """
        if origin_time <= 0:
            raise ValueError(f"origin_time must be positive, got {origin_time}")

        if not inplace:
            tree = tree.copy()

        if rescale:
            tree, _, origin_time = self.rescale_tree(
                tree=tree,
                origin_time=origin_time,
                inplace=True,
            )

        if feature_names is None:
            features_to_add = self.feature_names
        else:
            unknown_features = set(feature_names) - self.available_features
            if unknown_features:
                raise ValueError(
                    f"Unknown feature names: {unknown_features}. "
                    f"Available features: {self.available_features}"
                )
            features_to_add = list(feature_names)

        root = tree.get_tree_root()

        for node in tree.traverse(self.traversal_strategy):
            context = {
                "node": node,
                "root": root,
                "origin_time": origin_time,
            }
            for feature_name in features_to_add:
                self._feature_registry[feature_name](context)

        return tree

    def _ensure_feature(self, context: dict, feature_name: str) -> None:
        """
        Ensure that a dependent feature exists on the current node.

        This helper is used internally by built-in feature functions to lazily
        compute prerequisites.
        """
        node = context["node"]
        if not hasattr(node, feature_name):
            self._feature_registry[feature_name](context)

    def _is_close_to_zero(self, value: float) -> bool:
        """Return True if `value` is numerically close to zero."""
        return abs(value) <= self.time_tolerance

    def _is_close(self, a: float, b: float) -> bool:
        """Return True if `a` and `b` are numerically close."""
        return abs(a - b) <= self.time_tolerance

    def _add_node_time(self, context: dict) -> None:
        """
        Add `node_time` to the current node.

        Definition
        ----------
            node_time = origin_time - distance(root, node)

        Boundary handling
        -----------------
        - If node_time is within tolerance of 0, store 0.0
        - If node_time is within tolerance of origin_time, store origin_time
        """
        node = context["node"]
        root = context["root"]
        origin_time = context["origin_time"]

        node_root_distance = root.get_distance(node)
        node_time = origin_time - node_root_distance

        if self._is_close_to_zero(node_time):
            node_time = 0.0
        elif self._is_close(node_time, origin_time):
            node_time = origin_time

        node.add_feature("node_time", node_time)

    def _add_time_bin(self, context: dict) -> None:
        """
        Add `time_bin` to the current node.

        Dependency
        ----------
        Requires `node_time`, which is computed automatically if absent.

        Range
        -----
        `time_bin` is guaranteed to lie in:
            [0, num_time_bins - 1]
        """
        node = context["node"]
        origin_time = context["origin_time"]

        self._ensure_feature(context, "node_time")
        time_bin = self._calculate_time_bin(node.node_time, origin_time)
        node.add_feature("time_bin", time_bin)

    def _add_is_internal(self, context: dict) -> None:
        """
        Add `is_internal` to the current node.

        Definition
        ----------
        - 1 if the node is not a leaf
        - 0 otherwise
        """
        node = context["node"]
        node.add_feature("is_internal", 0 if node.is_leaf() else 1)

    def _add_is_tip(self, context: dict) -> None:
        """
        Add `is_tip` to the current node.

        Definition
        ----------
        - 1 if the node is a leaf
        - 0 otherwise
        """
        node = context["node"]
        node.add_feature("is_tip", 1 if node.is_leaf() else 0)

    def _add_branch_length(self, context: dict) -> None:
        """
        Add `branch_length` to the current node.

        Definition
        ----------
            branch_length = float(node.dist)

        Interpretation
        --------------
        Length of the edge connecting the node to its parent.
        """
        node = context["node"]
        node.add_feature("branch_length", float(node.dist))

    def _add_extant_sampling_probability(self, context: dict) -> None:
        """
        Add `extant_sampling_probability` to the current node.

        Current definition
        ------------------
        A constant scalar equal to `self.extant_sampling_probability`, assigned
        uniformly to all nodes.
        """
        node = context["node"]
        node.add_feature(
            "extant_sampling_probability",
            float(self.extant_sampling_probability),
        )

    def _add_is_fossil(self, context: dict) -> None:
        """
        Add `is_fossil` to the current node.

        Definition
        ----------
        If node is a leaf:
        - 0 if node_time is effectively 0
        - 1 otherwise

        If node is not a leaf:
        - 0

        Interpretation
        --------------
        Under the current implementation, only leaf nodes can be classified as
        fossil. Internal nodes are always assigned 0.
        """
        node = context["node"]
        self._ensure_feature(context, "node_time")

        if node.is_leaf():
            is_fossil = 0 if self._is_close_to_zero(node.node_time) else 1
        else:
            is_fossil = 0

        node.add_feature("is_fossil", is_fossil)

    def _add_is_extant(self, context: dict) -> None:
        """
        Add `is_extant` to the current node.

        Definition
        ----------
        If node is a leaf:
        - 1 if node_time is effectively 0
        - 0 otherwise

        If node is not a leaf:
        - 0

        Interpretation
        --------------
        Under the current implementation, only leaf nodes can be classified as
        extant.
        """
        node = context["node"]
        self._ensure_feature(context, "node_time")

        if node.is_leaf():
            is_extant = 1 if self._is_close_to_zero(node.node_time) else 0
        else:
            is_extant = 0

        node.add_feature("is_extant", is_extant)

    def _add_is_sampled_ancestor(self, context: dict) -> None:
        """
        Add `is_sampled_ancestor` to the current node.

        Dependency
        ----------
        Requires `is_fossil`, which is computed automatically if absent.

        Definition
        ----------
        If is_fossil == 1:
        - 1 if branch length is effectively 0
        - 0 otherwise

        Else:
        - 0

        Important note
        --------------
        This is the current implementation's operational definition. Whether it
        matches a given phylogenetic convention should be evaluated separately.
        """
        node = context["node"]
        self._ensure_feature(context, "is_fossil")

        if node.is_fossil == 1:
            is_sampled_ancestor = 1 if self._is_close_to_zero(node.dist) else 0
        else:
            is_sampled_ancestor = 0

        node.add_feature("is_sampled_ancestor", is_sampled_ancestor)

    def _add_is_not_sampled_ancestor(self, context: dict) -> None:
        """
        Add `is_not_sampled_ancestor` to the current node.

        Dependency
        ----------
        Requires `is_fossil`, which is computed automatically if absent.

        Definition
        ----------
        If is_fossil == 1:
        - 0 if branch length is effectively 0
        - 1 otherwise

        Else:
        - 0
        """
        node = context["node"]
        self._ensure_feature(context, "is_fossil")

        if node.is_fossil == 1:
            is_not_sampled_ancestor = 0 if self._is_close_to_zero(node.dist) else 1
        else:
            is_not_sampled_ancestor = 0

        node.add_feature("is_not_sampled_ancestor", is_not_sampled_ancestor)

    def _calculate_time_bin(self, node_time: float, origin_time: float) -> int:
        """
        Convert continuous `node_time` to a discrete time bin.

        Definition
        ----------
        The result is an integer in `[0, num_time_bins - 1]`.

        Rules:
        - If node_time is effectively <= 0:
              return 0
        - If node_time is effectively >= origin_time:
              return num_time_bins - 1
        - Otherwise:
              return ceil(node_time * (num_time_bins - 1) / origin_time)

        Boundary guarantee
        ------------------
        The returned bin is always clamped to:
            [0, num_time_bins - 1]
        """
        if self._is_close_to_zero(node_time) or node_time < 0:
            return 0
        if self._is_close(node_time, origin_time) or node_time > origin_time:
            return self.num_time_bins - 1

        time_bin = math.ceil(node_time * (self.num_time_bins - 1) / origin_time)
        return max(0, min(self.num_time_bins - 1, time_bin))

    def _validate_parameters(
        self,
        num_time_bins: int,
        extant_sampling_probability: float,
        traversal_strategy: str,
        time_tolerance: float,
    ) -> None:
        """Validate initialization parameters."""
        if num_time_bins < 2:
            raise ValueError(f"num_time_bins must be at least 2, got {num_time_bins}")

        if not 0.0 <= extant_sampling_probability <= 1.0:
            raise ValueError(
                f"extant_sampling_probability must be in [0, 1], "
                f"got {extant_sampling_probability}"
            )

        if traversal_strategy not in self.VALID_TRAVERSALS:
            raise ValueError(
                f"Invalid traversal_strategy='{traversal_strategy}'. "
                f"Valid options: {sorted(self.VALID_TRAVERSALS)}"
            )

        if time_tolerance < 0:
            raise ValueError(
                f"time_tolerance must be non-negative, got {time_tolerance}"
            )

    def __repr__(self) -> str:
        return (
            "TreeFeatureEngineer("
            f"num_time_bins={self.num_time_bins}, "
            f"extant_sampling_probability={self.extant_sampling_probability}, "
            f"traversal_strategy='{self.traversal_strategy}', "
            f"num_features={len(self.feature_names)})"
        )
