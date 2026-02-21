"""
Tree Feature Engineering Module

This module provides functionality to add features/attributes to ETE Tree nodes.
These features can then be used by the converter to create graph data.
"""

import math
from typing import Optional, List, Callable, Set, Tuple
from ete3 import Tree

class TreeFeatureEngineer:
    """Feature engineer for adding attributes to phylogenetic tree nodes

    This class adds computed features as attributes to each node in an ETE Tree.
    The tree can then be converted to graph data by TreeToGraphConverter.

    Attributes:
        num_time_bins: Number of time bins for discretizing the timeline
        extant_sampling_probability: Probability of sampling extant species
        available_features: Set of all available feature names
        custom_features: Dictionary of custom feature functions
        
    Example:
        >>> engineer = TreeFeatureEngineer(num_time_bins=101)
        >>> tree = Tree("((A:1,B:2)C:3,D:4)E;")
        >>> # Add all features with rescaling
        >>> tree_with_features = engineer.add_features(
        ...     tree, origin_time=10.0, rescale=True
        ... )
        >>> # Or rescale separately
        >>> tree, scale_factor, new_origin = engineer.rescale_tree(tree, origin_time=10.0)
    """

    def __init__(
        self,
        num_time_bins: int = 101,
        extant_sampling_probability: float = 1.0,
        custom_features: Optional[dict] = None
    ):
        """Initialize the feature engineer
        
        Args:
            num_time_bins: Number of time bins (default: 101)
            extant_sampling_probability: Sampling probability for extant species (default: 1.0)
            custom_features: Optional dict mapping feature names to feature functions (default: None)
        
        Raises:
            ValueError: If parameters are invalid
        """
        self._validate_parameters(num_time_bins, extant_sampling_probability)
        
        self.num_time_bins = num_time_bins
        self.extant_sampling_probability = extant_sampling_probability
        
        # Register all built-in features
        self._feature_registry = {
            'node_time': self._add_node_time,
            'time_bin': self._add_time_bin,
            'is_internal': self._add_is_internal,
            'is_tip': self._add_is_tip,
            'is_fossil': self._add_is_fossil,
            'is_extant': self._add_is_extant,
            'is_sampled_ancestor': self._add_is_sampled_ancestor,
            'is_not_sampled_ancestor': self._add_is_not_sampled_ancestor,
            'branch_length': self._add_branch_length,
            'extant_sampling_probability': self._add_extant_sampling_probability
        }
        
        # Add custom features to registry
        if custom_features:
            self._feature_registry.update(custom_features)
        
        self.available_features = set(self._feature_registry.keys())
    
    def rescale_tree(
        self,
        tree: Tree,
        origin_time: float,
        inplace: bool = True
    ) -> Tuple[Tree, float, float]:
        """Rescale tree so that mean of non-zero branch lengths equals 1
        
        Args:
            tree: ETE Tree object
            origin_time: Origin time (root age) of the tree before rescaling
            inplace: If True, modify tree in place; if False, copy first (default: True)
        
        Returns:
            Tuple of (rescaled_tree, scale_factor, new_origin_time)
            - rescaled_tree: Tree with rescaled branch lengths
            - scale_factor: The scaling factor applied (new_value = old_value * scale_factor)
            - new_origin_time: The rescaled origin time
        
        Example:
            >>> engineer = TreeFeatureEngineer()
            >>> tree = Tree("((A:1,B:2)C:3,D:4)E;")
            >>> rescaled_tree, factor, new_origin = engineer.rescale_tree(tree, origin_time=10.0)
            >>> print(f"Scale factor: {factor}, New origin: {new_origin}")
        """
        # Copy tree if not inplace
        if not inplace:
            tree = tree.copy()
        
        # Collect all non-zero branch lengths
        non_zero_lengths = []
        for node in tree.traverse():
            if node.dist > 0:
                non_zero_lengths.append(node.dist)
        
        if not non_zero_lengths:
            raise ValueError("Tree has no non-zero branch lengths to rescale")
        
        # Calculate mean of non-zero branch lengths
        mean_length = sum(non_zero_lengths) / len(non_zero_lengths)
        
        # Calculate scale factor
        scale_factor = 1.0 / mean_length
        
        # Rescale all branch lengths
        for node in tree.traverse():
            node.dist = node.dist * scale_factor
        
        # Rescale origin time
        new_origin_time = origin_time * scale_factor
        
        return tree, scale_factor, new_origin_time
        
    def add_features(
        self, 
        tree: Tree, 
        origin_time: float, 
        feature_names: Optional[List[str]] = None,
        rescale: bool = True,
        inplace: bool = True
    ) -> Tree:
        """Add features as attributes to all nodes in the tree
        
        Args:
            tree: ETE Tree object
            origin_time: Origin time (root age) of the tree
            feature_names: List of feature names to add. If None, adds all available features (default: None)
            rescale: If True, rescale tree before adding features (default: True)
            inplace: If True, modify tree in place; if False, copy first (default: True)
        
        Returns:
            Tree: Tree with features added as node attributes
        
        Raises:
            ValueError: If origin_time <= 0 or if unknown feature names are provided
        
        Example:
            >>> engineer = TreeFeatureEngineer(num_time_bins=50)
            >>> tree = Tree("((A:1,B:2)C:3,D:4)E;")
            >>> # Add all features with rescaling
            >>> tree_with_features = engineer.add_features(
            ...     tree, origin_time=10.0, rescale=True
            ... )
            >>> # Add only specific features without rescaling
            >>> tree_with_features = engineer.add_features(
            ...     tree, origin_time=10.0,
            ...     feature_names=['node_time', 'is_tip'],
            ...     rescale=False
            ... )
        """
        if origin_time <= 0:
            raise ValueError(f"origin_time must be positive, got {origin_time}")
        
        # Copy tree if not inplace
        if not inplace:
            tree = tree.copy()
        
        # Rescale if requested
        if rescale:
            tree, scale_factor, origin_time = self.rescale_tree(tree, origin_time, inplace=True)
        
        # Determine which features to add
        if feature_names is None:
            features_to_add = list(self._feature_registry.keys())
        else:
            # Validate feature names
            unknown_features = set(feature_names) - self.available_features
            if unknown_features:
                raise ValueError(
                    f"Unknown feature names: {unknown_features}. "
                    f"Available features: {self.available_features}"
                )
            features_to_add = feature_names
        
        root = tree.get_tree_root()
        
        # Add features to each node
        for node in tree.traverse():
            # Create context dict for feature computation
            context = {
                'node': node,
                'root': root,
                'origin_time': origin_time
            }
            
            # Add each requested feature
            for feature_name in features_to_add:
                feature_fn = self._feature_registry[feature_name]
                feature_fn(context)
        
        return tree

    # Feature computation methods
 
    def _add_node_time(self, context: dict) -> None:
        """Add node_time feature"""
        node = context['node']
        root = context['root']
        origin_time = context['origin_time']
        
        node_root_distance = root.get_distance(node)
        node_time = origin_time - node_root_distance
        node.add_feature('node_time', node_time)
    
    def _add_time_bin(self, context: dict) -> None:
        """Add time_bin feature"""
        node = context['node']
        origin_time = context['origin_time']
        
        # Ensure node_time exists
        if not hasattr(node, 'node_time'):
            self._add_node_time(context)
        
        time_bin = self._calculate_time_bin(node.node_time, origin_time)
        node.add_feature('time_bin', time_bin)
    
    def _add_is_internal(self, context: dict) -> None:
        """Add is_internal feature"""
        node = context['node']
        is_internal = 0 if node.is_leaf() else 1
        node.add_feature('is_internal', is_internal)
    
    def _add_is_tip(self, context: dict) -> None:
        """Add is_tip feature"""
        node = context['node']
        is_tip = 1 if node.is_leaf() else 0
        node.add_feature('is_tip', is_tip)
    
    def _add_branch_length(self, context: dict) -> None:
        """Add branch_length feature"""
        node = context['node']
        node.add_feature('branch_length', node.dist)
    
    def _add_extant_sampling_probability(self, context: dict) -> None:
        """Add extant_sampling_probability feature"""
        node = context['node']
        node.add_feature('extant_sampling_probability', self.extant_sampling_probability)
    
    def _add_is_fossil(self, context: dict) -> None:
        """Add is_fossil feature"""
        node = context['node']
        
        # Ensure node_time exists
        if not hasattr(node, 'node_time'):
            self._add_node_time(context)
        
        if node.is_leaf():
            is_fossil = 0 if node.node_time == 0 else 1
        else:
            is_fossil = 0
        
        node.add_feature('is_fossil', is_fossil)
    
    def _add_is_extant(self, context: dict) -> None:
        """Add is_extant feature"""
        node = context['node']
        
        # Ensure node_time exists
        if not hasattr(node, 'node_time'):
            self._add_node_time(context)
        
        if node.is_leaf():
            is_extant = 1 if node.node_time == 0 else 0
        else:
            is_extant = 0
        
        node.add_feature('is_extant', is_extant)
    
    def _add_is_sampled_ancestor(self, context: dict) -> None:
        """Add is_sampled_ancestor feature"""
        node = context['node']
        
        # Ensure is_fossil exists
        if not hasattr(node, 'is_fossil'):
            self._add_is_fossil(context)
        
        if node.is_fossil == 1:
            is_sampled_ancestor = 1 if node.dist == 0 else 0
        else:
            is_sampled_ancestor = 0
        
        node.add_feature('is_sampled_ancestor', is_sampled_ancestor)
    
    def _add_is_not_sampled_ancestor(self, context: dict) -> None:
        """Add is_not_sampled_ancestor feature"""
        node = context['node']
        
        # Ensure is_fossil exists
        if not hasattr(node, 'is_fossil'):
            self._add_is_fossil(context)
        
        if node.is_fossil == 1:
            is_not_sampled_ancestor = 0 if node.dist == 0 else 1
        else:
            is_not_sampled_ancestor = 0
        
        node.add_feature('is_not_sampled_ancestor', is_not_sampled_ancestor)

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
            f"extant_sampling_probability={self.extant_sampling_probability}, "
            f"available_features={len(self.available_features)})"
        )