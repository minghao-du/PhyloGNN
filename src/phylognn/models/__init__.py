"""
Neural network models for phylogenetic tree analysis.

This module provides various GNN architectures for learning from phylogenetic
trees, including GAT-based models, LSTM hybrids, and multi-task learning models.
"""

from .layers import GATBlock, PositionalEncoding, ResidualGATStack, MLPHead
from .base import BasePhyloGNN, BaseGATNet
from .gat_lstm import GATBiLSTMNet
from .multitask import MultiTaskGATNet, TaskHead

__all__ = [
# Layers
'GATBlock',
'PositionalEncoding',
'ResidualGATStack',
'MLPHead',

# Base classes
'BasePhyloGNN',
'BaseGATNet',

# Models
'GATBiLSTMNet',
'MultiTaskGATNet',
'TaskHead',
]