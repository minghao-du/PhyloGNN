"""
Multi-task learning models for phylogenetic analysis.

This module implements models designed for simultaneous prediction of
multiple related tasks, such as estimating different evolutionary parameters
or performing classification and regression jointly.
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_scatter import scatter
from typing import List, Optional

from .base import BaseGATNet

class TaskHead(nn.Module):
    """
    Task-specific head with BiLSTM and FC layers.

    This module processes time-binned features through bidirectional LSTM
    layers followed by fully connected layers to produce task-specific predictions.

    Args:
        input_size: Input feature dimension
        hidden_size: LSTM hidden dimension
        num_layers: Number of LSTM layers to stack
        dropout_prob: Dropout probability
        fc_hidden_dim: Hidden dimension for final FC layers
        output_dim: Output dimension for this task
        
    Example:
        >>> head = TaskHead(
        ...     input_size=256,
        ...     hidden_size=128,
        ...     num_layers=2,
        ...     dropout_prob=0.2,
        ...     fc_hidden_dim=64,
        ...     output_dim=1
        ... )
        >>> x = torch.randn(32, 101, 256)  # [batch, seq_len, features]
        >>> out = head(x)  # Shape: [32, 1]
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout_prob: float,
        fc_hidden_dim: int,
        output_dim: int
    ):
        super(TaskHead, self).__init__()
        
        # First BiLSTM layer
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.layernorm1 = nn.LayerNorm(hidden_size * 2)
        self.dropout1 = nn.Dropout(dropout_prob)
        
        # Second BiLSTM layer
        self.lstm2 = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.layernorm2 = nn.LayerNorm(hidden_size * 2)
        self.dropout2 = nn.Dropout(dropout_prob)
        
        # Final FC layers
        self.fc1 = nn.Linear(hidden_size * 2, fc_hidden_dim)
        self.fc2 = nn.Linear(fc_hidden_dim, output_dim)
        self.dropout_fc = nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the task head.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_size]
            
        Returns:
            Task predictions of shape [batch_size, output_dim]
        """
        # First LSTM
        x, _ = self.lstm1(x)
        x = self.layernorm1(x)
        x = self.dropout1(x)
        
        # Second LSTM
        x, _ = self.lstm2(x)
        x = self.layernorm2(x)
        x = self.dropout2(x)
        
        # Average pooling over sequence
        x = x.mean(dim=1)
        
        # FC layers
        x = torch.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        
        return x