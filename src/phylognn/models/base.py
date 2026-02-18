"""
Base model classes for phylogenetic GNN architectures.

This module provides abstract base classes and common functionality
for building phylogenetic tree analysis models.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional
from torch_geometric.data import Data

from .layers import ResidualGATStack

class BasePhyloGNN(nn.Module, ABC):
    """
    Abstract base class for phylogenetic GNN models.

    This class defines the common interface and shared functionality
    for all phylogenetic GNN architectures in the package.

    Subclasses must implement:
        - forward(): Main forward pass logic
        - get_embedding_dim(): Return the dimension of learned embeddings
    """
    
    def __init__(self):
        super(BasePhyloGNN, self).__init__()
    
    
    @abstractmethod
    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            data: PyTorch Geometric Data object containing:
                - x: Node features [num_nodes, num_features]
                - edge_index: Graph connectivity [2, num_edges]
                - batch: Batch assignment [num_nodes]
                
        Returns:
            Model predictions
        """
        pass
    
    @abstractmethod
    def get_embedding_dim(self) -> int:
        """
        Get the dimension of the learned node embeddings.
        
        Returns:
            Embedding dimension
        """
        pass
    
    def get_num_parameters(self) -> int:
        """
        Count the total number of trainable parameters.
        
        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_encoder(self):
        """
        Freeze the encoder parameters for transfer learning.
        
        This is useful when you want to fine-tune only the prediction head
        while keeping the learned representations fixed.
        """
        for name, param in self.named_parameters():
            if 'fc' not in name and 'head' not in name:
                param.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = True

class BaseGATNet(BasePhyloGNN):
    """
    Base GAT network with configurable architecture.

    This class provides a foundation for GAT-based models with a standard
    structure: preprocessing -> GAT layers -> task-specific head.

    Args:
        input_dim: Input feature dimension
        preprocess_dim: Dimension after preprocessing layer
        gat_hidden_dim: Hidden dimension for GAT layers
        gat_heads: Number of attention heads
        num_gat_layers: Number of GAT layers
        dropout_prob: Dropout probability
        use_preprocessing: Whether to use preprocessing layer
        
    Example:
        >>> # This is a base class, typically used through subclasses
        >>> class MyGATModel(BaseGATNet):
        ...     def forward(self, data):
        ...         x = self.preprocess(data.x) if self.use_preprocessing else data.x
        ...         x = self.gat_encoder(x, data.edge_index)
        ...         # Add task-specific logic here
        ...         return x
    """
    
    def __init__(
        self,
        input_dim: int,
        preprocess_dim: int,
        gat_hidden_dim: int,
        gat_heads: int = 4,
        num_gat_layers: int = 3,
        dropout_prob: float = 0.2,
        use_preprocessing: bool = True
    ):
        super(BaseGATNet, self).__init__()
        
        self.use_preprocessing = use_preprocessing
        self.gat_hidden_dim = gat_hidden_dim
        self.gat_heads = gat_heads
        
        # Preprocessing layer
        if use_preprocessing:
            self.preprocess = nn.Sequential(
                nn.Linear(input_dim, preprocess_dim),
                nn.ReLU()
            )
            gat_input_dim = preprocess_dim
        else:
            self.preprocess = nn.Identity()
            gat_input_dim = input_dim
        
        # GAT encoder with residual connections
        self.gat_encoder = ResidualGATStack(
            in_channels=gat_input_dim,
            hidden_channels=gat_hidden_dim,
            num_layers=num_gat_layers,
            heads=gat_heads,
            dropout_prob=dropout_prob
        )
        
        self.dropout = nn.Dropout(dropout_prob)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Encode node features using preprocessing and GAT layers.
        
        Args:
            x: Node feature matrix [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            
        Returns:
            Encoded node features [num_nodes, gat_hidden_dim * gat_heads]
        """
        if self.use_preprocessing:
            x = self.preprocess(x)
        x = self.gat_encoder(x, edge_index)
        return x

    def get_embedding_dim(self) -> int:
        """Get the dimension of GAT output embeddings."""
        return self.gat_hidden_dim * self.gat_heads

    @abstractmethod
    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass - must be implemented by subclasses."""
        pass