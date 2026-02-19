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
    
class MultiTaskGATNet(BaseGATNet):
    """
    Multi-task GAT network for joint prediction of multiple phylogenetic parameters.

    This model uses a shared GAT encoder to learn common representations,
    then branches into task-specific heads for predicting different parameters.
    This approach enables knowledge sharing across related tasks while maintaining
    task-specific specialization.

    Args:
        input_dim: Input feature dimension
        task_configs: List of dictionaries, each containing:
            - 'name': Task name (str)
            - 'output_dim': Output dimension for this task (int)
            - 'lstm_hidden_dim': LSTM hidden dimension (int, default: 128)
            - 'fc_hidden_dim': FC hidden dimension (int, default: 64)
        preprocess_fc_dim: Preprocessing layer dimension
        gat_hidden_dim: GAT hidden dimension per head
        gat_heads: Number of GAT attention heads
        num_gat_layers: Number of GAT layers
        num_lstm_layers: Number of LSTM layers per task head
        dropout_prob: Dropout probability
        
    Example:
        >>> task_configs = [
        ...     {'name': 'speciation_rate', 'output_dim': 1},
        ...     {'name': 'extinction_rate', 'output_dim': 1},
        ...     {'name': 'sampling_prob', 'output_dim': 1}
        ... ]
        >>> model = MultiTaskGATNet(
        ...     input_dim=4,
        ...     task_configs=task_configs,
        ...     gat_hidden_dim=64,
        ...     gat_heads=4
        ... )
        >>> outputs = model(data)  # Returns dict with keys: task names
    """
        
    def __init__(
        self,
        input_dim: int,
        task_configs: List[Dict],
        preprocess_fc_dim: int = 32,
        gat_hidden_dim: int = 64,
        gat_heads: int = 4,
        num_gat_layers: int = 3,
        num_lstm_layers: int = 2,
        dropout_prob: float = 0.2
    ):
        super(MultiTaskGATNet, self).__init__(
            input_dim=input_dim,
            preprocess_dim=preprocess_fc_dim,
            gat_hidden_dim=gat_hidden_dim,
            gat_heads=gat_heads,
            num_gat_layers=num_gat_layers,
            dropout_prob=dropout_prob,
            use_preprocessing=(preprocess_fc_dim != 1)
        )
        
        self.task_names = [config['name'] for config in task_configs]
        
        # Input size for LSTM: GAT output + time bin label
        lstm_input_size = gat_hidden_dim * gat_heads + 1
        
        # Layer normalization for LSTM input
        self.layernorm_lstm_input = nn.LayerNorm(lstm_input_size)
        
        # Create task-specific heads
        self.task_heads = nn.ModuleDict()
        for config in task_configs:
            task_name = config['name']
            output_dim = config['output_dim']
            lstm_hidden = config.get('lstm_hidden_dim', 128)
            fc_hidden = config.get('fc_hidden_dim', 64)
            
            self.task_heads[task_name] = TaskHead(
                input_size=lstm_input_size,
                hidden_size=lstm_hidden,
                num_layers=num_lstm_layers,
                dropout_prob=dropout_prob,
                fc_hidden_dim=fc_hidden,
                output_dim=output_dim
            )

    def forward(self, data: Data) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the multi-task network.
        
        Args:
            data: PyTorch Geometric Data object with:
                - x: Node features [num_nodes, input_dim]
                    x[:, 1] should contain time bin labels
                - edge_index: Graph connectivity [2, num_edges]
                - batch: Batch assignment [num_nodes]
                
        Returns:
            Dictionary mapping task names to predictions
            Each prediction has shape [batch_size, task_output_dim]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Shared GAT encoding
        x = self.encode(x, edge_index)
        
        # Pool by time bins
        pooled = self._pool_by_time_bins(x, data.x, batch)
        
        # Normalize LSTM input
        pooled = self.layernorm_lstm_input(pooled)
        
        # Task-specific predictions
        outputs = {}
        for task_name, task_head in self.task_heads.items():
            outputs[task_name] = task_head(pooled)
        
        return outputs

    def _pool_by_time_bins(
        self,
        x: torch.Tensor,
        original_features: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Pool node features by time bins.
        
        Args:
            x: Encoded node features [num_nodes, feature_dim]
            original_features: Original node features containing time bin labels
            batch: Batch assignment [num_nodes]
            
        Returns:
            Pooled features [batch_size, num_bins, feature_dim + 1]
        """
        batch_size = int(batch.max().item()) + 1
        group_labels = original_features[:, 1].long()  # Time bin labels
        num_groups = int(group_labels.max().item()) + 1
        
        # Create unique index for (batch, time_bin) pairs
        group_index = batch * num_groups + group_labels
        
        # Aggregate features by (batch, time_bin)
        pooled = scatter(
            x, group_index, dim=0,
            dim_size=batch_size * num_groups,
            reduce='sum'
        )
        pooled = pooled.view(batch_size, num_groups, -1)
        
        # Add time bin labels as features
        time_bin_labels = torch.arange(
            num_groups, device=pooled.device
        ).unsqueeze(0).unsqueeze(2).float()
        time_bin_labels = time_bin_labels.expand(batch_size, -1, -1)
        
        pooled = torch.cat([pooled, time_bin_labels], dim=2)
        return pooled