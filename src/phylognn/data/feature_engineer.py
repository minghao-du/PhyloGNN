"""
Feature engineering utilities for phylogenetic trees.

This module provides TreeFeatureEngineer, which adds computed attributes
to ETE Tree nodes for downstream graph conversion and machine learning.
"""

from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple
import math

from ete3 import Tree


FeatureFunction = Callable[[dict], None]


class TreeFeatureEngineer:
    """Feature engineer for adding attributes to ETE Tree nodes.

    Parameters
    ----------
    num_time_bins : int, default=101
        Number of time bins used for time discretization.
    extant_sampling_probability : float, default=1.0
        Sampling probability for extant species.
    custom_features : Optional[Dict[str, FeatureFunction]], default=None
        Additional custom feature functions to register.
    traversal_strategy : str, default="preorder"
        ETE traversal strategy used when adding features.
    time_tolerance : float, default=1e-8
        Numerical tolerance used when comparing time values to zero/origin time.

    Notes
    -----
    Built-in features:
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
        """Rescale tree so that the mean of non-zero branch lengths equals 1.

        Parameters
        ----------
        tree : Tree
            Input ETE tree.
        origin_time : float
            Original tree origin time.
        inplace : bool, default=True
            Whether to modify the input tree in place.

        Returns
        -------
        Tuple[Tree, float, float]
            (rescaled_tree, scale_factor, new_origin_time)
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
        """Add requested features as node attributes.

        Parameters
        ----------
        tree : Tree
            Input ETE tree.
        origin_time : float
            Root age / origin time.
        feature_names : Optional[Sequence[str]], default=None
            Requested features. If None, all registered features are added.
        rescale : bool, default=True
            Whether to rescale the tree before feature computation.
        inplace : bool, default=True
            Whether to modify the input tree in place.

        Returns
        -------
        Tree
            Tree with node features attached.
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
        """Ensure a dependent feature exists on the current node."""
        node = context["node"]
        if not hasattr(node, feature_name):
            self._feature_registry[feature_name](context)

    def _is_close_to_zero(self, value: float) -> bool:
        return abs(value) <= self.time_tolerance

    def _is_close(self, a: float, b: float) -> bool:
        return abs(a - b) <= self.time_tolerance

    def _add_node_time(self, context: dict) -> None:
        """Add node_time feature."""
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
        """Add time_bin feature."""
        node = context["node"]
        origin_time = context["origin_time"]

        self._ensure_feature(context, "node_time")
        time_bin = self._calculate_time_bin(node.node_time, origin_time)
        node.add_feature("time_bin", time_bin)

    def _add_is_internal(self, context: dict) -> None:
        """Add is_internal feature."""
        node = context["node"]
        node.add_feature("is_internal", 0 if node.is_leaf() else 1)

    def _add_is_tip(self, context: dict) -> None:
        """Add is_tip feature."""
        node = context["node"]
        node.add_feature("is_tip", 1 if node.is_leaf() else 0)

    def _add_branch_length(self, context: dict) -> None:
        """Add branch_length feature."""
        node = context["node"]
        node.add_feature("branch_length", float(node.dist))

    def _add_extant_sampling_probability(self, context: dict) -> None:
        """Add extant_sampling_probability feature."""
        node = context["node"]
        node.add_feature(
            "extant_sampling_probability",
            float(self.extant_sampling_probability),
        )

    def _add_is_fossil(self, context: dict) -> None:
        """Add is_fossil feature."""
        node = context["node"]
        self._ensure_feature(context, "node_time")

        if node.is_leaf():
            is_fossil = 0 if self._is_close_to_zero(node.node_time) else 1
        else:
            is_fossil = 0

        node.add_feature("is_fossil", is_fossil)

    def _add_is_extant(self, context: dict) -> None:
        """Add is_extant feature."""
        node = context["node"]
        self._ensure_feature(context, "node_time")

        if node.is_leaf():
            is_extant = 1 if self._is_close_to_zero(node.node_time) else 0
        else:
            is_extant = 0

        node.add_feature("is_extant", is_extant)

    def _add_is_sampled_ancestor(self, context: dict) -> None:
        """Add is_sampled_ancestor feature."""
        node = context["node"]
        self._ensure_feature(context, "is_fossil")

        if node.is_fossil == 1:
            is_sampled_ancestor = 1 if self._is_close_to_zero(node.dist) else 0
        else:
            is_sampled_ancestor = 0

        node.add_feature("is_sampled_ancestor", is_sampled_ancestor)

    def _add_is_not_sampled_ancestor(self, context: dict) -> None:
        """Add is_not_sampled_ancestor feature."""
        node = context["node"]
        self._ensure_feature(context, "is_fossil")

        if node.is_fossil == 1:
            is_not_sampled_ancestor = 0 if self._is_close_to_zero(node.dist) else 1
        else:
            is_not_sampled_ancestor = 0

        node.add_feature("is_not_sampled_ancestor", is_not_sampled_ancestor)

    def _calculate_time_bin(self, node_time: float, origin_time: float) -> int:
        """Convert continuous node time to a discrete time bin."""
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
