"""
Neural network layer components for phylogenetic GNN models.

This module provides reusable building blocks including graph attention layers,
positional encodings, and other architectural components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from typing import Optional

class GATBlock(nn.Module):
    """
    Graph Attention Network block with batch normalization and dropout.

    This block combines a GAT convolution layer with batch normalization
    and dropout for improved training stability and regularization.

    Args:
        in_channels: Number of input features per node
        out_channels: Number of output features per node (per attention head)
        heads: Number of attention heads
        dropout_prob: Dropout probability
        concat: Whether to concatenate or average attention head outputs
        negative_slope: Negative slope for LeakyReLU in attention mechanism
        
    Example:
        >>> block = GATBlock(in_channels=64, out_channels=32, heads=4, dropout_prob=0.2)
        >>> x = torch.randn(100, 64)  # 100 nodes, 64 features
        >>> edge_index = torch.randint(0, 100, (2, 500))  # 500 edges
        >>> out = block(x, edge_index)  # Shape: [100, 128] (32 * 4 heads)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        dropout_prob: float = 0.2,
        concat: bool = True,
        negative_slope: float = 0.2
    ):
        super(GATBlock, self).__init__()
        
        self.gat = GATConv(
            in_channels,
            out_channels,
            heads=heads,
            dropout=dropout_prob,
            concat=concat,
            negative_slope=negative_slope
        )
        
        # Batch normalization dimension depends on whether heads are concatenated
        bn_dim = out_channels * heads if concat else out_channels
        self.bn = nn.BatchNorm1d(bn_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the GAT block.
        
        Args:
            x: Node feature matrix of shape [num_nodes, in_channels]
            edge_index: Graph connectivity in COO format of shape [2, num_edges]
            
        Returns:
            Node embeddings of shape [num_nodes, out_channels * heads]
        """
        x = F.elu(self.gat(x, edge_index))
        x = self.bn(x)
        x = self.dropout(x)
        return x

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for sequence data.

    Adds position-dependent patterns to input embeddings using sine and cosine
    functions of different frequencies, as introduced in "Attention is All You Need".

    Args:
        d_model: Dimension of the model embeddings
        dropout: Dropout probability
        max_len: Maximum sequence length to pre-compute encodings for
        
    Example:
        >>> pe = PositionalEncoding(d_model=128, dropout=0.1, max_len=1000)
        >>> x = torch.randn(32, 100, 128)  # [batch, seq_len, d_model]
        >>> x_with_pos = pe(x)
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute positional encodings once in log space
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # Even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd dimensions
        pe = pe.unsqueeze(0)  # Add batch dimension: [1, max_len, d_model]
        
        # Register as buffer (not a parameter, but part of state_dict)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            Tensor with positional encoding added, same shape as input
        """
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return self.dropout(x)