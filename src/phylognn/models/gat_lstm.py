"""
GAT-LSTM hybrid models for phylogenetic tree analysis.

This module implements models that combine Graph Attention Networks (GAT)
for spatial feature learning with Bidirectional LSTMs for temporal/sequential
pattern recognition in phylogenetic trees.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
from torch_geometric.data import Data
from torch_scatter import scatter
from typing import Optional, Literal

from .base import BaseGATNet
from .layers import MLPHead

class GATBiLSTMNet(BaseGATNet):
    """
    GAT-BiLSTM network for phylogenetic parameter estimation.

    This model combines Graph Attention Networks for learning node-level
    representations with Bidirectional LSTMs for capturing temporal patterns
    in time-binned phylogenetic data.

    Architecture:
        1. Optional preprocessing layer
        2. Stacked GAT layers with residual connections
        3. Time-bin pooling and aggregation
        4. Bidirectional LSTM layers (or FC layers as alternative)
        5. Final prediction head

    Args:
        input_dim: Input feature dimension
        output_dim: Output dimension (number of parameters to predict)
        preprocess_fc_dim: Preprocessing layer hidden dimension (1 to disable)
        gat_hidden_dim: GAT hidden dimension per head
        lstm_hidden_dim: LSTM hidden dimension (1 to disable LSTM)
        output_final_fc_dim: Final FC layer hidden dimension
        gat_heads: Number of GAT attention heads
        dropout_prob: Dropout probability
        num_lstm_layers: Number of LSTM layers (0 to use FC instead)
        output_positive: Whether to enforce positive outputs with ReLU
        num_gat_layers: Number of GAT layers
        
    Example:
        >>> model = GATBiLSTMNet(
        ...     input_dim=4,
        ...     output_dim=2,
        ...     preprocess_fc_dim=32,
        ...     gat_hidden_dim=64,
        ...     lstm_hidden_dim=128,
        ...     output_final_fc_dim=64,
        ...     gat_heads=4,
        ...     dropout_prob=0.2,
        ...     num_lstm_layers=2
        ... )
        >>> # data.x should have shape [num_nodes, 4]
        >>> # data.x[:, 1] should contain time bin labels
        >>> output = model(data)  # Shape: [batch_size, 2]
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        preprocess_fc_dim: int = 32,
        gat_hidden_dim: int = 64,
        lstm_hidden_dim: int = 128,
        output_final_fc_dim: int = 64,
        gat_heads: int = 4,
        dropout_prob: float = 0.2,
        num_lstm_layers: int = 2,
        output_positive: bool = False,
        num_gat_layers: int = 3
    ):
        # Determine GAT input dimension based on preprocessing
        use_preprocessing = (preprocess_fc_dim != 1)
        gat_input_dim = preprocess_fc_dim if use_preprocessing else input_dim
        
        super(GATBiLSTMNet, self).__init__(
            input_dim=input_dim,
            preprocess_dim=preprocess_fc_dim,
            gat_hidden_dim=gat_hidden_dim,
            gat_heads=gat_heads,
            num_gat_layers=num_gat_layers,
            dropout_prob=dropout_prob,
            use_preprocessing=use_preprocessing
        )
        
        # Configuration flags
        self.add_pooling_lstm = (lstm_hidden_dim != 1)
        self.use_fc_instead_of_lstm = (num_lstm_layers == 0)
        self.output_positive = output_positive
        
        # Determine final FC input dimension
        if not self.add_pooling_lstm:
            # Direct pooling without LSTM
            fc_input_dim = gat_hidden_dim * gat_heads
        elif self.use_fc_instead_of_lstm:
            # Use FC layers instead of LSTM
            fc_input_dim = lstm_hidden_dim * 2
        else:
            # Use LSTM
            fc_input_dim = lstm_hidden_dim * 2  # Bidirectional
        
        # Build architecture based on configuration
        if self.add_pooling_lstm:
            if self.use_fc_instead_of_lstm:
                # FC-based pooling (alternative to LSTM)
                self._build_fc_pooling(
                    gat_hidden_dim, gat_heads, lstm_hidden_dim, dropout_prob
                )
            else:
                # LSTM-based temporal modeling
                self._build_lstm_layers(
                    gat_hidden_dim, gat_heads, lstm_hidden_dim,
                    num_lstm_layers, dropout_prob
                )
        
        # Final prediction head
        self.fc = MLPHead(
            input_dim=fc_input_dim,
            hidden_dim=output_final_fc_dim,
            output_dim=output_dim,
            num_hidden_layers=1,
            dropout_prob=dropout_prob,
            output_activation='relu' if output_positive else None
        )

    def _build_fc_pooling(
        self,
        gat_hidden_dim: int,
        gat_heads: int,
        lstm_hidden_dim: int,
        dropout_prob: float
    ):
        """Build FC layers for pooling (alternative to LSTM)."""
        # Assuming 101 time bins (can be made configurable)
        input_size = 101 * (gat_hidden_dim * gat_heads + 1)
        
        self.fc_after_gat_1 = nn.Sequential(
            nn.Linear(input_size, lstm_hidden_dim * 4),
            nn.ReLU(),
            nn.BatchNorm1d(lstm_hidden_dim * 4),
            nn.Dropout(dropout_prob)
        )
        
        self.fc_after_gat_2 = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 4, lstm_hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(lstm_hidden_dim * 2),
            nn.Dropout(dropout_prob)
        )

    def _build_lstm_layers(
        self,
        gat_hidden_dim: int,
        gat_heads: int,
        lstm_hidden_dim: int,
        num_lstm_layers: int,
        dropout_prob: float
    ):
        """Build LSTM layers for temporal modeling."""
        lstm_input_size = gat_hidden_dim * gat_heads + 1  # +1 for time bin label
        
        # Input normalization
        self.layernorm_lstm_input = nn.LayerNorm(lstm_input_size)
        
        # First LSTM layer
        self.lstm1 = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True
        )
        self.layernorm_lstm1 = nn.LayerNorm(lstm_hidden_dim * 2)
        
        # Second LSTM layer
        self.lstm2 = nn.LSTM(
            input_size=lstm_hidden_dim * 2,
            hidden_size=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True
        )
        self.layernorm_lstm2 = nn.LayerNorm(lstm_hidden_dim * 2)

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass through the GAT-BiLSTM network.
        
        Args:
            data: PyTorch Geometric Data object with:
                - x: Node features [num_nodes, input_dim]
                    x[:, 1] should contain time bin labels (0 to num_bins-1)
                - edge_index: Graph connectivity [2, num_edges]
                - batch: Batch assignment [num_nodes]
                
        Returns:
            Predictions of shape [batch_size, output_dim]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Encode with GAT
        x = self.encode(x, edge_index)
        
        if self.add_pooling_lstm:
            # Pool nodes by time bins
            pooled = self._pool_by_time_bins(x, data.x, batch)
            
            if self.use_fc_instead_of_lstm:
                # FC-based processing
                pooled_flat = pooled.view(pooled.size(0), -1)
                out = self.fc_after_gat_1(pooled_flat)
                out = self.fc_after_gat_2(out)
            else:
                # LSTM-based processing
                out = self._process_with_lstm(pooled)
        else:
            # Direct global pooling
            out = global_add_pool(x, batch)
        
        # Final prediction
        out = self.fc(out)
        return out

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

    def _process_with_lstm(self, pooled: torch.Tensor) -> torch.Tensor:
        """
        Process time-binned features with BiLSTM.
        
        Args:
            pooled: Pooled features [batch_size, num_bins, feature_dim]
            
        Returns:
            Processed features [batch_size, lstm_hidden_dim * 2]
        """
        # Normalize input
        pooled = self.layernorm_lstm_input(pooled)
        
        # First LSTM layer
        lstm_out, _ = self.lstm1(pooled)
        lstm_out = self.layernorm_lstm1(lstm_out)
        lstm_out = self.dropout(lstm_out)
        
        # Second LSTM layer
        lstm_out, _ = self.lstm2(lstm_out)
        lstm_out = self.layernorm_lstm2(lstm_out)
        lstm_out = self.dropout(lstm_out)
        
        # Average pooling over time dimension
        lstm_out = lstm_out.mean(dim=1)
        return lstm_out