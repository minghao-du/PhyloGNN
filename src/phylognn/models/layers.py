"""
Reusable neural network building blocks for phylogenetic GNN models.

This module provides:
    - graph attention blocks
    - plain GAT stacks
    - residual GAT stacks
    - sinusoidal positional encoding
    - bidirectional LSTM temporal sequence encoding
    - flexible MLP heads

Design principles:
    - each custom module exposes an explicit `reset_parameters()` where possible
    - docstrings specify tensor shapes and computation order
    - building blocks are reusable across multiple downstream models
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

GATEncoderType = Literal["gat", "res_gat"]
TemporalAggregation = Literal["mean", "last", "max"]


def validate_positive_int_counters(**counters: object) -> None:
    """Require model counters to be positive Python ints, excluding bool."""
    invalid_type_names = sorted(name for name, value in counters.items() if type(value) is not int)
    invalid_range_names = sorted(
        name for name, value in counters.items() if type(value) is int and value < 1
    )

    if invalid_type_names:
        details = ["invalid type for " + ", ".join(f"`{name}`" for name in invalid_type_names)]
        if invalid_range_names:
            details.append(
                "non-positive value for " + ", ".join(f"`{name}`" for name in invalid_range_names)
            )
        raise TypeError(
            "Counter settings must be positive non-bool Python integers; "
            + "; ".join(details)
            + "."
        )

    if invalid_range_names:
        raise ValueError(
            "Counter settings must be positive non-bool Python integers; non-positive value for "
            + ", ".join(f"`{name}`" for name in invalid_range_names)
            + "."
        )


class GATBlock(nn.Module):
    """
    Graph Attention block with activation, normalization, and dropout.

    Computation order:
        GATConv -> ELU -> BatchNorm1d -> Dropout

    Args:
        in_channels:
            Number of input node features.
        out_channels:
            Number of output features per attention head.
        heads:
            Number of attention heads.
        dropout_prob:
            Dropout probability used inside GATConv and after normalization.
        concat:
            Whether attention head outputs are concatenated.
        negative_slope:
            Negative slope used in GAT attention mechanism.

    Input:
        - x: [num_nodes, in_channels]
        - edge_index: [2, num_edges]

    Output:
        - [num_nodes, out_channels * heads] if `concat=True`
        - [num_nodes, out_channels] if `concat=False`
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        dropout_prob: float = 0.2,
        concat: bool = True,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError(f"`in_channels` must be > 0, got {in_channels}.")
        if out_channels <= 0:
            raise ValueError(f"`out_channels` must be > 0, got {out_channels}.")
        if heads <= 0:
            raise ValueError(f"`heads` must be > 0, got {heads}.")
        if not (0.0 <= dropout_prob < 1.0):
            raise ValueError(f"`dropout_prob` must be in [0, 1), got {dropout_prob}.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout_prob = dropout_prob
        self.negative_slope = negative_slope

        self.gat = GATConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            dropout=dropout_prob,
            concat=concat,
            negative_slope=negative_slope,
        )

        bn_dim = out_channels * heads if concat else out_channels
        self.bn = nn.BatchNorm1d(bn_dim)
        self.dropout = nn.Dropout(dropout_prob)

    @property
    def output_dim(self) -> int:
        """
        Effective output feature dimension.

        Returns:
            Output channel dimension after head concatenation/aggregation.
        """
        return self.out_channels * self.heads if self.concat else self.out_channels

    def reset_parameters(self) -> None:
        """
        Reset all learnable parameters in the block.
        """
        self.gat.reset_parameters()
        self.bn.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Apply graph attention, activation, normalization, and dropout.

        Args:
            x:
                Node feature matrix of shape [num_nodes, in_channels].
            edge_index:
                Graph connectivity of shape [2, num_edges].

        Returns:
            Node embeddings of shape [num_nodes, output_dim].
        """
        x = self.gat(x, edge_index)
        x = F.elu(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"heads={self.heads}, "
            f"concat={self.concat}, "
            f"dropout_prob={self.dropout_prob}"
        )


class GATStack(nn.Module):
    """
    Plain stack of GAT blocks without residual connections.

    Architecture:
        x -> GATBlock_1 -> GATBlock_2 -> ... -> GATBlock_L

    Args:
        in_channels:
            Input node feature dimension.
        hidden_channels:
            Output dimension per attention head.
        num_layers:
            Number of stacked GAT blocks. Must be a positive non-bool Python
            integer.
        heads:
            Number of attention heads per block.
        dropout_prob:
            Dropout probability used in each block.

    Output dimension:
        `hidden_channels * heads`
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 3,
        heads: int = 4,
        dropout_prob: float = 0.2,
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError(f"`in_channels` must be > 0, got {in_channels}.")
        if hidden_channels <= 0:
            raise ValueError(f"`hidden_channels` must be > 0, got {hidden_channels}.")
        validate_positive_int_counters(num_layers=num_layers)
        if heads <= 0:
            raise ValueError(f"`heads` must be > 0, got {heads}.")

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.heads = heads
        self.dropout_prob = dropout_prob
        self.output_dim = hidden_channels * heads

        self.blocks = nn.ModuleList()
        current_dim = in_channels

        for _ in range(num_layers):
            self.blocks.append(
                GATBlock(
                    in_channels=current_dim,
                    out_channels=hidden_channels,
                    heads=heads,
                    dropout_prob=dropout_prob,
                    concat=True,
                )
            )
            current_dim = self.output_dim

    def reset_parameters(self) -> None:
        """
        Reset all GAT blocks.
        """
        for block in self.blocks:
            block.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Apply stacked GAT blocks without residual connections.

        Args:
            x:
                Node features [num_nodes, in_channels].
            edge_index:
                Graph connectivity [2, num_edges].

        Returns:
            Node embeddings [num_nodes, hidden_channels * heads].
        """
        for block in self.blocks:
            x = block(x, edge_index)
        return x

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, "
            f"hidden_channels={self.hidden_channels}, "
            f"num_layers={self.num_layers}, "
            f"heads={self.heads}, "
            f"dropout_prob={self.dropout_prob}"
        )


class ResidualGATBlock(nn.Module):
    """
    Residual graph attention block.

    Structure:
        residual projection(x) + GATBlock(x, edge_index)

    A learnable linear projection is used when the input dimension differs from
    the GAT block output dimension.

    Args:
        in_channels:
            Input node feature dimension.
        out_channels:
            Output feature dimension per GAT head.
        heads:
            Number of attention heads.
        dropout_prob:
            Dropout probability used inside the GAT block.

    Input:
        - x: [num_nodes, in_channels]
        - edge_index: [2, num_edges]

    Output:
        - [num_nodes, out_channels * heads]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        dropout_prob: float = 0.2,
    ) -> None:
        super().__init__()

        self.gat_block = GATBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            dropout_prob=dropout_prob,
            concat=True,
        )
        self.output_dim = out_channels * heads

        if in_channels == self.output_dim:
            self.residual_proj = nn.Identity()
        else:
            self.residual_proj = nn.Linear(in_channels, self.output_dim)

    def reset_parameters(self) -> None:
        """
        Reset GAT and residual projection parameters.
        """
        self.gat_block.reset_parameters()
        if hasattr(self.residual_proj, "reset_parameters"):
            self.residual_proj.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Apply residual GAT transformation.

        Args:
            x:
                Node features [num_nodes, in_channels].
            edge_index:
                Graph connectivity [2, num_edges].

        Returns:
            Node features [num_nodes, output_dim].
        """
        return self.gat_block(x, edge_index) + self.residual_proj(x)


class ResidualGATStack(nn.Module):
    """
    Stack of residual GAT blocks.

    Architecture:
        x
          -> ResidualGATBlock_1
          -> ResidualGATBlock_2
          -> ...
          -> ResidualGATBlock_L

    The first block may project from `in_channels` to `hidden_channels * heads`.
    Subsequent blocks keep a constant embedding dimension.

    Args:
        in_channels:
            Input node feature dimension.
        hidden_channels:
            Hidden dimension per attention head.
        num_layers:
            Number of residual GAT blocks. Must be a positive non-bool Python
            integer.
        heads:
            Number of attention heads.
        dropout_prob:
            Dropout probability used in each block.

    Output dimension:
        `hidden_channels * heads`
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 3,
        heads: int = 4,
        dropout_prob: float = 0.2,
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError(f"`in_channels` must be > 0, got {in_channels}.")
        if hidden_channels <= 0:
            raise ValueError(f"`hidden_channels` must be > 0, got {hidden_channels}.")
        validate_positive_int_counters(num_layers=num_layers)
        if heads <= 0:
            raise ValueError(f"`heads` must be > 0, got {heads}.")

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.heads = heads
        self.dropout_prob = dropout_prob
        self.output_dim = hidden_channels * heads

        self.blocks = nn.ModuleList()
        current_dim = in_channels

        for _ in range(num_layers):
            block = ResidualGATBlock(
                in_channels=current_dim,
                out_channels=hidden_channels,
                heads=heads,
                dropout_prob=dropout_prob,
            )
            self.blocks.append(block)
            current_dim = self.output_dim

    def reset_parameters(self) -> None:
        """
        Reset all residual GAT blocks.
        """
        for block in self.blocks:
            block.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Apply stacked residual GAT blocks.

        Args:
            x:
                Node features [num_nodes, in_channels].
            edge_index:
                Graph connectivity [2, num_edges].

        Returns:
            Node embeddings [num_nodes, hidden_channels * heads].
        """
        for block in self.blocks:
            x = block(x, edge_index)
        return x

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, "
            f"hidden_channels={self.hidden_channels}, "
            f"num_layers={self.num_layers}, "
            f"heads={self.heads}, "
            f"dropout_prob={self.dropout_prob}"
        )


def build_gat_encoder(
    encoder_type: GATEncoderType,
    in_channels: int,
    hidden_channels: int,
    num_layers: int,
    heads: int,
    dropout_prob: float,
) -> nn.Module:
    """
    Build a GAT encoder stack.

    Args:
        encoder_type:
            Encoder variant:
                - "gat": plain stacked GAT blocks
                - "res_gat": residual GAT stack
        in_channels:
            Input node feature dimension.
        hidden_channels:
            Hidden dimension per attention head.
        num_layers:
            Number of GAT layers.
        heads:
            Number of attention heads.
        dropout_prob:
            Dropout probability.

    Returns:
        A GAT encoder module.

    Raises:
        ValueError:
            If `encoder_type` is unsupported.
    """
    if encoder_type == "gat":
        return GATStack(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            heads=heads,
            dropout_prob=dropout_prob,
        )
    if encoder_type == "res_gat":
        return ResidualGATStack(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            heads=heads,
            dropout_prob=dropout_prob,
        )

    raise ValueError(
        f"Unsupported `encoder_type`: {encoder_type!r}. " "Expected one of ('gat', 'res_gat')."
    )


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for sequence-style inputs.

    This module is independent from graph topology and is intended for
    sequence/temporal models that consume tensors of shape
    [batch_size, seq_len, d_model].

    Args:
        d_model:
            Feature dimension of the sequence embeddings.
        dropout:
           out probability applied after adding positional encodings.
        max_len:
            Maximum supported sequence length.

    Input:
        - x: [batch_size, seq_len, d_model]

    Output:
        - same shape as input
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()

        if d_model <= 0:
            raise ValueError(f"`d_model` must be > 0, got {d_model}.")
        if max_len <= 0:
            raise ValueError(f"`max_len` must be > 0, got {max_len}.")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"`dropout` must be in [0, 1), got {dropout}.")

        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)

        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional information to sequence embeddings.

        Args:
            x:
                Input tensor [batch_size, seq_len, d_model].

        Returns:
            Tensor of same shape as input.

        Raises:
            ValueError:
                If input sequence length exceeds `max_len` or shape is invalid.
        """
        if x.dim() != 3:
            raise ValueError(
                f"`x` must be 3D [batch_size, seq_len, d_model], got shape {tuple(x.shape)}."
            )
        if x.size(1) > self.max_len:
            raise ValueError(f"Input sequence length {x.size(1)} exceeds max_len={self.max_len}.")
        if x.size(2) != self.d_model:
            raise ValueError(
                f"Input feature dim must equal d_model={self.d_model}, got {x.size(2)}."
            )

        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TemporalBiLSTMEncoder(nn.Module):
    """
    Reusable bidirectional LSTM encoder for prepared temporal sequences.

    This layer is independent from PyG graph metadata. Graph-specific pooling
    into time bins must happen before calling this encoder.

    Processing order:
        LayerNorm(input) -> BiLSTM -> LayerNorm(output) -> Dropout -> aggregation

    Args:
        input_dim:
            Feature dimension of each sequence step.
        hidden_dim:
            Hidden dimension for one LSTM direction.
        num_layers:
            Number of stacked recurrent layers. Must be a positive non-bool
            Python integer.
        dropout_prob:
            Dropout probability applied to recurrent outputs. The LSTM-internal
            dropout follows PyTorch semantics and is active only when
            `num_layers > 1`.
        aggregation:
            Temporal aggregation rule:
                - "mean": average recurrent outputs across sequence steps
                - "last": select the final recurrent output
                - "max": take the maximum across sequence steps

    Input:
        - sequence: [batch_size, seq_len, input_dim]

    Output:
        - [batch_size, hidden_dim * 2]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout_prob: float = 0.2,
        aggregation: TemporalAggregation = "mean",
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError(f"`input_dim` must be > 0, got {input_dim}.")
        if hidden_dim <= 0:
            raise ValueError(f"`hidden_dim` must be > 0, got {hidden_dim}.")
        validate_positive_int_counters(num_layers=num_layers)
        if not (0.0 <= dropout_prob < 1.0):
            raise ValueError(f"`dropout_prob` must be in [0, 1), got {dropout_prob}.")
        if aggregation not in {"mean", "last", "max"}:
            raise ValueError(
                f"`aggregation` must be one of ('mean', 'last', 'max'), got {aggregation!r}."
            )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        self.aggregation = aggregation
        self.output_dim = hidden_dim * 2

        self.input_norm = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_prob if num_layers > 1 else 0.0,
        )
        self.output_norm = nn.LayerNorm(self.output_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def reset_parameters(self) -> None:
        """
        Reset all learnable parameters in the encoder.
        """
        self.input_norm.reset_parameters()
        self.lstm.reset_parameters()
        self.output_norm.reset_parameters()

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Encode a prepared temporal sequence.

        Args:
            sequence:
                Tensor of shape [batch_size, seq_len, input_dim].

        Returns:
            Tensor of shape [batch_size, hidden_dim * 2].

        Raises:
            TypeError:
                If `sequence` is not a torch tensor.
            ValueError:
                If rank, sequence length, or feature dimension is invalid.
        """
        if not torch.is_tensor(sequence):
            raise TypeError("`sequence` must be a torch.Tensor.")
        if sequence.dim() != 3:
            raise ValueError(
                "`sequence` must be 3D [batch_size, seq_len, input_dim], "
                f"got shape {tuple(sequence.shape)}."
            )
        if sequence.size(1) <= 0:
            raise ValueError("`sequence` length must be > 0.")
        if sequence.size(2) != self.input_dim:
            raise ValueError(
                f"`sequence` feature dim must equal input_dim={self.input_dim}, "
                f"got {sequence.size(2)}."
            )

        sequence = self.input_norm(sequence)
        recurrent_output, _ = self.lstm(sequence)
        recurrent_output = self.output_norm(recurrent_output)
        recurrent_output = self.dropout(recurrent_output)
        return self._aggregate(recurrent_output)

    def _aggregate(self, recurrent_output: torch.Tensor) -> torch.Tensor:
        if self.aggregation == "mean":
            return recurrent_output.mean(dim=1)
        if self.aggregation == "last":
            return recurrent_output[:, -1, :]
        if self.aggregation == "max":
            return recurrent_output.max(dim=1).values

        raise RuntimeError(f"Unsupported aggregation={self.aggregation!r}.")

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"num_layers={self.num_layers}, "
            f"dropout_prob={self.dropout_prob}, "
            f"aggregation={self.aggregation!r}"
        )


class MLPHead(nn.Module):
    """
    Flexible MLP prediction head.

    Architecture:
        [Linear -> ReLU -> LayerNorm -> Dropout] * num_hidden_layers
        -> Linear(output)

    LayerNorm keeps graph-level prediction heads valid for training batches
    containing a single graph.

    Optional output activation can be applied at the final layer.

    Args:
        input_dim:
            Input feature dimension.
        hidden_dim:
            Hidden layer dimension. Required if `num_hidden_layers > 0`.
        output_dim:
            Output feature dimension.
        num_hidden_layers:
            Number of hidden layers before the output layer.
        dropout_prob:
            Dropout probability in hidden layers.
        output_activation:
            Optional output activation:
                - "relu"
                - "sigmoid"
                - "tanh"
                - None

    Input:
        - x: [batch_size, input_dim]

    Output:
        - [batch_size, output_dim]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int],
        output_dim: int,
        num_hidden_layers: int = 1,
        dropout_prob: float = 0.2,
        output_activation: Optional[str] = None,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError(f"`input_dim` must be > 0, got {input_dim}.")
        if output_dim <= 0:
            raise ValueError(f"`output_dim` must be > 0, got {output_dim}.")
        if num_hidden_layers < 0:
            raise ValueError(f"`num_hidden_layers` must be >= 0, got {num_hidden_layers}.")
        if not (0.0 <= dropout_prob < 1.0):
            raise ValueError(f"`dropout_prob` must be in [0, 1), got {dropout_prob}.")
        if num_hidden_layers > 0:
            if hidden_dim is None or hidden_dim <= 0:
                raise ValueError(
                    "`hidden_dim` must be provided and > 0 when " "`num_hidden_layers > 0`."
                )

        valid_activations = {None, "relu", "sigmoid", "tanh"}
        if output_activation not in valid_activations:
            raise ValueError(
                f"`output_activation` must be one of {valid_activations}, "
                f"got {output_activation!r}."
            )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_hidden_layers = num_hidden_layers
        self.dropout_prob = dropout_prob
        self.output_activation = output_activation

        layers = []
        current_dim = input_dim

        for _ in range(num_hidden_layers):
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_prob),
                ]
            )
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, output_dim))

        if output_activation == "relu":
            layers.append(nn.ReLU())
        elif output_activation == "sigmoid":
            layers.append(nn.Sigmoid())
        elif output_activation == "tanh":
            layers.append(nn.Tanh())

        self.mlp = nn.Sequential(*layers)

    def reset_parameters(self) -> None:
        """
        Reset all resettable submodules in the MLP head.
        """
        for module in self.mlp:
            if hasattr(module, "reset_parameters") and callable(module.reset_parameters):
                module.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the MLP head.

        Args:
            x:
                Input tensor [batch_size, input_dim].

        Returns:
            Output tensor [batch_size, output_dim].
        """
        return self.mlp(x)

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"output_dim={self.output_dim}, "
            f"num_hidden_layers={self.num_hidden_layers}, "
            f"dropout_prob={self.dropout_prob}, "
            f"output_activation={self.output_activation}"
        )
