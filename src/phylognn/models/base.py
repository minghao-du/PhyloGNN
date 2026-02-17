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