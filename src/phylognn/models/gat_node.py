"""
GAT model for node-level predictions.

This module implements a model that uses Graph Attention Networks (GAT) to generate
per-node representations, followed by a final MLP prediction head without any
global pooling.
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn as nn
from torch_geometric.data import Data

from .base import BaseGATNet, GATEncoderType, _reset_module_parameters
from .layers import MLPHead


class GATNodeRegressor(BaseGATNet):
    """
    GAT-based graph model for node-level regression.

    High-level architecture:
        1. Optional feature preprocessing
        2. GAT encoder backbone (plain or residual)
        3. MLP prediction head applied to each node embedding independently

    Required input fields:
        - data.x: FloatTensor [num_nodes, input_dim]
        - data.edge_index: LongTensor [2, num_edges]
        - data.batch: Optional LongTensor [num_nodes] (not required for node regression)

    Args:
        input_dim:
            Input node feature dimension.
        output_dim:
            Prediction dimension per node.
        preprocess_dim:
            Optional preprocessing projection dimension. Ignored if
            `use_preprocessing=False`.
        gat_hidden_dim:
            GAT hidden dimension per attention head.
        gat_heads:
            Number of attention heads.
        num_gat_layers:
            Number of GAT blocks in the encoder. Must be a positive non-bool
            Python integer.
        dropout_prob:
            Dropout probability used across encoder and temporal modules.
        use_preprocessing:
            Whether to apply a learnable preprocessing layer before GAT.
        encoder_type:
            Type of GAT encoder backbone:
                - "gat"
                - "res_gat"
        head_hidden_dim:
            Hidden dimension of the final MLP prediction head.

    Output:
        Tensor of shape [num_nodes, output_dim].
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        preprocess_dim: Optional[int] = 32,
        gat_hidden_dim: int = 64,
        gat_heads: int = 4,
        num_gat_layers: int = 3,
        dropout_prob: float = 0.2,
        use_preprocessing: bool = True,
        encoder_type: GATEncoderType = "res_gat",
        head_hidden_dim: int = 64,
    ) -> None:
        if output_dim <= 0:
            raise ValueError(f"`output_dim` must be > 0, got {output_dim}.")
        if head_hidden_dim <= 0:
            raise ValueError(f"`head_hidden_dim` must be > 0, got {head_hidden_dim}.")

        super().__init__(
            input_dim=input_dim,
            preprocess_dim=preprocess_dim,
            gat_hidden_dim=gat_hidden_dim,
            gat_heads=gat_heads,
            num_gat_layers=num_gat_layers,
            dropout_prob=dropout_prob,
            use_preprocessing=use_preprocessing,
            encoder_type=encoder_type,
        )

        self.output_dim = output_dim
        self.head_hidden_dim = head_hidden_dim

        encoder_dim = self.get_embedding_dim()

        self.head = MLPHead(
            input_dim=encoder_dim,
            hidden_dim=head_hidden_dim,
            output_dim=output_dim,
            num_hidden_layers=1,
            dropout_prob=dropout_prob,
            output_activation=None,
        )

        self.reset_parameters()

    def get_head_modules(self) -> Iterable[nn.Module]:
        """
        Return task-specific prediction head modules.

        Returns:
            Iterable containing the prediction head.
        """
        return [self.head]

    def reset_parameters(self) -> None:
        """
        Reset all model parameters.

        Reset order:
            1. base GAT encoder
            2. prediction head
        """
        super().reset_parameters()
        _reset_module_parameters(self.head)

    def forward(self, data: Data) -> torch.Tensor:
        """
        Run the full forward pass.

        Required fields:
            Always required:
                - data.x: [num_nodes, input_dim]
                - data.edge_index: [2, num_edges]

        Shape contract:
            - node encoder output: [num_nodes, embedding_dim]
            - final output: [num_nodes, output_dim]

        Args:
            data:
                PyG Data object.

        Returns:
            Prediction tensor of shape [num_nodes, output_dim].
        """
        x = self.encode_data(data)
        return self.head(x)

    def extra_repr(self) -> str:
        """
        Extra representation shown in `print(model)`.
        """
        return f"output_dim={self.output_dim}, " f"head_hidden_dim={self.head_hidden_dim}"
