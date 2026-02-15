"""
Tree Feature Engineering Module

This module provides functionality to add features/attributes to ETE Tree nodes.
These features can then be used by the converter to create graph data.
"""

import math
from typing import Optional, List, Callable
from ete3 import Tree

class TreeFeatureEngineer:
    """Feature engineer for adding attributes to phylogenetic tree nodes

    This class adds computed features as attributes to each node in an ETE Tree.
    The tree can then be converted to graph data by TreeToGraphConverter.

    Attributes:
        num_time_bins: Number of time bins for discretizing the timeline
        extant_sampling_probability: Probability of sampling extant species
        feature_names: List of feature names that will be added to nodes

    Example:
        >>> engineer = TreeFeatureEngineer(num_time_bins=101)
        >>> tree = Tree("((A:1,B:2)C:3,D:4)E;")
        >>> tree_with_features = engineer.add_features(tree, origin_time=10.0)
        >>> # Now tree nodes have attributes like node.node_time, node.time_bin, etc.
    """

    def __init__(
        self,
        num_time_bins: int = 101,
        extant_sampling_probability: float = 1.0,
        custom_features: Optional[List[Callable]] = None
    ):
        """Initialize the feature engineer
        
        Args:
            num_time_bins: Number of time bins (default: 101)
            extant_sampling_probability: Sampling probability for extant species (default: 1.0)
            custom_features: Optional list of custom feature functions (default: None)
        
        Raises:
            ValueError: If parameters are invalid
        """
        self._validate_parameters(num_time_bins, extant_sampling_probability)
        
        self.num_time_bins = num_time_bins
        self.extant_sampling_probability = extant_sampling_probability
        self.custom_features = custom_features or []
        
        # Standard features that will be added
        self.feature_names = [
            'node_time',
            'time_bin',
            'is_internal',
            'is_tip',
            'is_fossil',
            'is_extant',
            'is_sampled_ancestor',
            'is_not_sampled_ancestor',
            'branch_length',
            'extant_sampling_probability'
        ]
        
    def add_features(self, tree: Tree, origin_time: float, inplace: bool = True) -> Tree:
        """Add features as attributes to all nodes in the tree
        
        Args:
            tree: ETE Tree object
            origin_time: Origin time (root age) of the tree
            inplace: If True, modify tree in place; if False, copy first (default: True)
        
        Returns:
            Tree: Tree with features added as node attributes
        
        Raises:
            ValueError: If origin_time <= 0
        
        Example:
            >>> engineer = TreeFeatureEngineer(num_time_bins=50)
            >>> tree = Tree("((A:1,B:2)C:3,D:4)E;")
            >>> tree_with_features = engineer.add_features(tree, origin_time=10.0)
            >>> print(tree_with_features.get_tree_root().node_time)
            10.0
        """
        if origin_time <= 0:
            raise ValueError(f"origin_time must be positive, got {origin_time}")
        
        # Copy tree if not inplace
        if not inplace:
            tree = tree.copy()
        
        root = tree.get_tree_root()
        
        # Add features to each node
        for node in tree.traverse():
            self._add_node_features(node, root, origin_time)
            
            # Add custom features if provided
            for custom_feature_fn in self.custom_features:
                custom_feature_fn(node, root, origin_time)
        
        return tree
    
    def _add_node_features(self, node: Tree, root: Tree, origin_time: float) -> None:
        """Add standard features to a single node
        
        Args:
            node: Node to add features to
            root: Root node of the tree
            origin_time: Origin time of the tree
        """
        # Calculate node time
        node_root_distance = root.get_distance(node)
        node_time = origin_time - node_root_distance
        node.add_feature('node_time', node_time)
        
        # Calculate time bin
        time_bin = self._calculate_time_bin(node_time, origin_time)
        node.add_feature('time_bin', time_bin)
        
        # Node type
        is_leaf = node.is_leaf()
        node.add_feature('is_internal', 0 if is_leaf else 1)
        node.add_feature('is_tip', 1 if is_leaf else 0)
        
        # Branch length
        node.add_feature('branch_length', node.dist)
        
        # Extant sampling probability
        node.add_feature('extant_sampling_probability', self.extant_sampling_probability)
        
        # Fossil/extant status
        if is_leaf:
            if node_time == 0:
                # Extant species
                node.add_feature('is_fossil', 0)
                node.add_feature('is_extant', 1)
                is_fossil_node = False
            else:
                # Fossil
                node.add_feature('is_fossil', 1)
                node.add_feature('is_extant', 0)
                is_fossil_node = True
        else:
            # Internal node
            node.add_feature('is_fossil', 0)
            node.add_feature('is_extant', 0)
            is_fossil_node = False
        
        # Sampled ancestor status
        if is_fossil_node:
            if node.dist == 0:
                node.add_feature('is_sampled_ancestor', 1)
                node.add_feature('is_not_sampled_ancestor', 0)
            else:
                node.add_feature('is_sampled_ancestor', 0)
                node.add_feature('is_not_sampled_ancestor', 1)
        else:
            node.add_feature('is_sampled_ancestor', 0)
            node.add_feature('is_not_sampled_ancestor', 0)

    # TODO: fixed number of time bins? or fixed time bin width? 
    # TODO: Tree average?
    def _calculate_time_bin(self, node_time: float, origin_time: float) -> int:
        """Calculate time bin index for a node
        
        Args:
            node_time: Time of the node
            origin_time: Origin time of the tree
        
        Returns:
            Time bin index (0 to num_time_bins-1)
        """
        if node_time <= 0:
            return 0
        elif node_time >= origin_time:
            return self.num_time_bins - 1
        else:
            time_bin = math.ceil(node_time * (self.num_time_bins - 1) / origin_time)
            return max(0, min(self.num_time_bins - 1, time_bin))
    
    def _validate_parameters(
        self,
        num_time_bins: int,
        extant_sampling_probability: float
    ) -> None:
        """Validate initialization parameters"""
        if not 0.0 <= extant_sampling_probability <= 1.0:
            raise ValueError(
                f"extant_sampling_probability must be in [0, 1], "
                f"got {extant_sampling_probability}"
            )
        
        if num_time_bins < 2:
            raise ValueError(
                f"num_time_bins must be at least 2, got {num_time_bins}"
            )
    
    def __repr__(self) -> str:
        return (
            f"TreeFeatureEngineer("
            f"num_time_bins={self.num_time_bins}, "
            f"extant_sampling_probability={self.extant_sampling_probability})"
        )