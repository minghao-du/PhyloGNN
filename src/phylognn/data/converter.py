"""
Tree to Graph Converter Module

This module converts ETE Tree objects (with node attributes) into
PyTorch Geometric Data objects.
"""

from typing import List, Optional, Dict
import torch
from torch_geometric.data import Data
from ete3 import Tree

class TreeToGraphConverter:
    """Converter for transforming trees with attributes into graph data

    This class reads attributes from ETE Tree nodes and converts the tree
    structure into a PyTorch Geometric Data object. It expects nodes to
    already have the necessary attributes (added by TreeFeatureEngineer or manually).

    Attributes:
        feature_names: List of node attribute names to extract as features
        add_virtual_nodes: Whether to add virtual nodes for time bins
        num_time_bins: Number of time bins (required if add_virtual_nodes=True)

    Example:
        >>> from phylognn.data import TreeFeatureEngineer, TreeToGraphConverter
        >>> 
        >>> # Step 1: Add features to tree
        >>> engineer = TreeFeatureEngineer(num_time_bins=101)
        >>> tree = Tree("((A:1,B:2)C:3,D:4)E;")
        >>> tree_with_features = engineer.add_features(tree, origin_time=10.0)
        >>> 
        >>> # Step 2: Convert to graph
        >>> converter = TreeToGraphConverter(feature_names=engineer.feature_names)
        >>> data = converter.convert(tree_with_features)
    """

    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        add_virtual_nodes: bool = True,
        num_time_bins: Optional[int] = None
    ):
        """Initialize the converter
        
        Args:
            feature_names: List of node attribute names to use as features.
                        If None, uses default features (default: None)
            add_virtual_nodes: Whether to add virtual nodes (default: True)
            num_time_bins: Number of time bins, required if add_virtual_nodes=True
        
        Raises:
            ValueError: If add_virtual_nodes=True but num_time_bins is None
        """
        if feature_names is None:
            # Default features (standard phylogenetic features)
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
        else:
            self.feature_names = feature_names
        
        self.add_virtual_nodes = add_virtual_nodes
        self.num_time_bins = num_time_bins
        
        if add_virtual_nodes and num_time_bins is None:
            raise ValueError("num_time_bins must be provided when add_virtual_nodes=True")

    def convert(self, tree: Tree) -> Data:
        """Convert an ETE Tree with attributes to PyTorch Geometric Data
        
        Args:
            tree: ETE Tree object with node attributes
        
        Returns:
            Data: PyTorch Geometric Data object with:
                - x: Node features [num_nodes, num_features]
                - edge_index: Edge connectivity [2, num_edges]
        
        Raises:
            AttributeError: If tree nodes are missing required attributes
        
        Example:
            >>> converter = TreeToGraphConverter(
            ...     feature_names=['node_time', 'time_bin', 'is_tip']
            ... )
            >>> data = converter.convert(tree_with_features)
            >>> print(data.x.shape)  # [num_nodes, 3]
        """
        # Extract features and build graph structure
        node_features, edge_index = self._extract_features_and_edges(tree)
        
        # Create Data object
        data = Data(x=node_features, edge_index=edge_index)
        
        # Add virtual nodes if configured
        if self.add_virtual_nodes:
            data = self._add_virtual_nodes(data)
        
        return data

    def convert_batch(self, trees: List[Tree]) -> List[Data]:
        """Convert multiple trees to graph data
        
        Args:
            trees: List of ETE Tree objects with attributes
        
        Returns:
            List of PyTorch Geometric Data objects
        
        Example:
            >>> trees = [tree1, tree2, tree3]  # All with features
            >>> data_list = converter.convert_batch(trees)
        """
        return [self.convert(tree) for tree in trees]

    def _extract_features_and_edges(self, tree: Tree) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract node features and edge indices from tree
        
        Args:
            tree: ETE Tree with node attributes
        
        Returns:
            Tuple of (node_features, edge_index)
        
        Raises:
            AttributeError: If nodes are missing required attributes
        """
        # Map nodes to indices
        node_to_idx: Dict[Tree, int] = {
            node: idx for idx, node in enumerate(tree.traverse())
        }
        
        # Extract features
        feature_matrix = []
        edge_index = []
        
        for node in tree.traverse():
            # Extract features for this node
            node_feature_vector = []
            for feature_name in self.feature_names:
                if not hasattr(node, feature_name):
                    raise AttributeError(
                        f"Node is missing required attribute '{feature_name}'. "
                        f"Did you forget to run TreeFeatureEngineer.add_features()?"
                    )
                node_feature_vector.append(getattr(node, feature_name))
            
            feature_matrix.append(node_feature_vector)
            
            # Build edges (bidirectional)
            for child in node.children:
                parent_idx = node_to_idx[node]
                child_idx = node_to_idx[child]
                edge_index.append([parent_idx, child_idx])
                edge_index.append([child_idx, parent_idx])
        
        # Convert to tensors
        features = torch.tensor(feature_matrix, dtype=torch.float32)
        edges = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.empty((2, 0), dtype=torch.long)
        
        return features, edges
    
    # TODO: More general virtual node addition that doesn't assume specific feature indices for time and time_bin
    def _add_virtual_nodes(self, data: Data) -> Data:
        """Add virtual nodes representing time bins
        
        Args:
            data: Original graph data
        
        Returns:
            Data: Graph with virtual nodes added
        """
        num_original_nodes = data.num_nodes
        num_features = data.x.size(1)
        
        # Create virtual node features
        virtual_features = torch.zeros((self.num_time_bins, num_features))
        
        # Set features for virtual nodes
        # Assumes feature_names[0] is time-related and feature_names[1] is time_bin
        for i in range(self.num_time_bins):
            virtual_features[i, 0] = float(i)  # time (using bin index)
            virtual_features[i, 1] = float(i)  # time_bin
            # Copy other features from first node if they exist
            if num_features > 9:
                virtual_features[i, 9] = data.x[0, 9].item()  # sampling_prob
        
        # Concatenate features
        data.x = torch.cat([data.x, virtual_features], dim=0)
        
        # Get time bins (assumes time_bin is feature index 1)
        original_time_bins = data.x[:num_original_nodes, 1]
        virtual_time_bins = data.x[num_original_nodes:, 1]
        
        new_edges = []
        
        # Connect virtual nodes to original nodes
        for virtual_idx, virtual_bin in enumerate(virtual_time_bins):
            matching_nodes = (original_time_bins == virtual_bin).nonzero(as_tuple=True)[0]
            virtual_node_idx = num_original_nodes + virtual_idx
            
            for node_idx in matching_nodes:
                new_edges.append([virtual_node_idx, node_idx.item()])
                new_edges.append([node_idx.item(), virtual_node_idx])
        
        # Connect adjacent virtual nodes
        virtual_node_indices = torch.arange(
            num_original_nodes,
            num_original_nodes + self.num_time_bins
        )
        
        for i in range(self.num_time_bins):
            current_bin = virtual_time_bins[i]
            current_idx = virtual_node_indices[i]
            
            neighbor_bin = current_bin + 1
            if 0 <= neighbor_bin < self.num_time_bins:
                neighbor_indices = (virtual_time_bins == neighbor_bin).nonzero(as_tuple=True)[0]
                neighbor_global_indices = virtual_node_indices[neighbor_indices]
                
                for neighbor_idx in neighbor_global_indices:
                    new_edges.append([current_idx.item(), neighbor_idx.item()])
                    new_edges.append([neighbor_idx.item(), current_idx.item()])
        
        # Add new edges
        if new_edges:
            new_edge_index = torch.tensor(new_edges, dtype=torch.long).t().contiguous()
            data.edge_index = torch.cat([data.edge_index, new_edge_index], dim=1)
        
        return data
    
    def __repr__(self) -> str:
        return (
            f"TreeToGraphConverter("
            f"num_features={len(self.feature_names)}, "
            f"add_virtual_nodes={self.add_virtual_nodes})"
        )    