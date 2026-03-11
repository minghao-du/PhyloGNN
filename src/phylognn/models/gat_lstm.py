"""
GAT-temporal hybrid models for phylogenetic tree analysis.

This module implements models that combine:
    - Graph Attention Networks (GAT) for node-level representation learning
    - Optional temporal aggregation over time bins
    - Optional FC or BiLSTM temporal encoders
    - A final MLP prediction head

Compared to earlier research-style implementations, this version makes the
data contract explicit:
    - time-bin labels are provided as `data.time_bin`
    - the number of time bins is configured explicitly
    - temporal behavior is controlled by a clear `temporal_mode`

Supported temporal modes:
    - "none": graph-level pooling directly from GAT node embeddings
    - "fc": time-bin pooling followed by flattening and FC temporal encoder
    - "lstm": time-bin pooling followed by bidirectional LSTM encoder
"""

from __future__ import annotations

from typing import Iterable, Literal, Optional

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from torch_scatter import scatter

from .base import BaseGATNet, GATEncoderType, _reset_module_parameters
from .layers import MLPHead

TemporalMode = Literal["none", "fc", "lstm"]
GraphPooling = Literal["sum", "mean", "max"]


class GATBiLSTMNet(BaseGATNet):
    """
    GAT-based graph model with optional temporal encoding over time bins.

    High-level architecture:
        1. Optional feature preprocessing
        2. GAT encoder backbone (plain or residual)
        3. One of:
            - direct graph pooling                     (temporal_mode="none")
            - time-bin pooling -> FC temporal encoder (temporal_mode="fc")
            - time-bin pooling -> BiLSTM encoder      (temporal_mode="lstm")
        4. MLP prediction head

    Required input fields:
        - data.x: FloatTensor [num_nodes, input_dim]
        - data.edge_index: LongTensor [2, num_edges]
        - data.batch: LongTensor [num_nodes]

    Additional required field when temporal_mode != "none":
        - data.time_bin: LongTensor [num_nodes], values in [0, num_time_bins - 1]

    Args:
        input_dim:
            Input node feature dimension.
        output_dim:
            Prediction dimension.
        preprocess_dim:
            Optional preprocessing projection dimension. Ignored if
            `use_preprocessing=False`.
        gat_hidden_dim:
            GAT hidden dimension per attention head.
        gat_heads:
            Number of attention heads.
        num_gat_layers:
            Number of GAT blocks in the encoder.
        dropout_prob:
            Dropout probability used across encoder and temporal modules.
        use_preprocessing:
            Whether to apply a learnable preprocessing layer before GAT.
        encoder_type:
            Type of GAT encoder backbone:
                - "gat"
                - "res_gat"
        temporal_mode:
            Temporal aggregation strategy: "none", "fc", or "lstm".
        num_time_bins:
            Total number of time bins. Required if temporal_mode != "none".
        temporal_hidden_dim:
            Hidden dimension used by the FC or LSTM temporal encoder.
        num_lstm_layers:
            Number of stacked recurrent layers in the BiLSTM encoder.
            Used only when `temporal_mode="lstm"`.
        graph_pool:
            Graph pooling mode used when `temporal_mode="none"`.
        head_hidden_dim:
            Hidden dimension of the final MLP prediction head.
        output_positive:
            If True, applies ReLU activation on the final output.

    Output:
        Tensor of shape [batch_size, output_dim].
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
        temporal_mode: TemporalMode = "lstm",
        num_time_bins: Optional[int] = None,
        temporal_hidden_dim: int = 128,
        num_lstm_layers: int = 2,
        graph_pool: GraphPooling = "sum",
        head_hidden_dim: int = 64,
        output_positive: bool = False,
    ) -> None:
        self._validate_model_args(
            output_dim=output_dim,
            temporal_mode=temporal_mode,
            num_time_bins=num_time_bins,
            temporal_hidden_dim=temporal_hidden_dim,
            num_lstm_layers=num_lstm_layers,
            graph_pool=graph_pool,
            head_hidden_dim=head_hidden_dim,
        )

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
        self.temporal_mode = temporal_mode
        self.num_time_bins = num_time_bins
        self.temporal_hidden_dim = temporal_hidden_dim
        self.num_lstm_layers = num_lstm_layers
        self.graph_pool = graph_pool
        self.head_hidden_dim = head_hidden_dim
        self.output_positive = output_positive

        encoder_dim = self.get_embedding_dim()

        if self.temporal_mode == "none":
            head_input_dim = encoder_dim

        elif self.temporal_mode == "fc":
            temporal_input_dim = self.num_time_bins * (encoder_dim + 1)
            self.temporal_mlp = nn.Sequential(
                nn.Linear(temporal_input_dim, temporal_hidden_dim * 4),
                nn.ReLU(),
                nn.BatchNorm1d(temporal_hidden_dim * 4),
                nn.Dropout(dropout_prob),
                nn.Linear(temporal_hidden_dim * 4, temporal_hidden_dim * 2),
                nn.ReLU(),
                nn.BatchNorm1d(temporal_hidden_dim * 2),
                nn.Dropout(dropout_prob),
            )
            head_input_dim = temporal_hidden_dim * 2

        elif self.temporal_mode == "lstm":
            lstm_input_dim = encoder_dim + 1
            self.temporal_input_norm = nn.LayerNorm(lstm_input_dim)
            self.temporal_lstm = nn.LSTM(
                input_size=lstm_input_dim,
                hidden_size=temporal_hidden_dim,
                num_layers=num_lstm_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout_prob if num_lstm_layers > 1 else 0.0,
            )
            self.temporal_output_norm = nn.LayerNorm(temporal_hidden_dim * 2)
            head_input_dim = temporal_hidden_dim * 2

        self.head = MLPHead(
            input_dim=head_input_dim,
            hidden_dim=head_hidden_dim,
            output_dim=output_dim,
            num_hidden_layers=1,
            dropout_prob=dropout_prob,
            output_activation="relu" if output_positive else None,
        )

        self.reset_parameters()

    @staticmethod
    def _validate_model_args(
        output_dim: int,
        temporal_mode: TemporalMode,
        num_time_bins: Optional[int],
        temporal_hidden_dim: int,
        num_lstm_layers: int,
        graph_pool: GraphPooling,
        head_hidden_dim: int,
    ) -> None:
        """
        Validate model-specific constructor arguments.

        Raises:
            ValueError:
                If any argument is invalid.
        """
        if output_dim <= 0:
            raise ValueError(f"`output_dim` must be > 0, got {output_dim}.")

        if temporal_mode not in {"none", "fc", "lstm"}:
            raise ValueError(
                f"`temporal_mode` must be one of ('none', 'fc', 'lstm'), got {temporal_mode!r}."
            )

        if temporal_mode != "none":
            if num_time_bins is None:
                raise ValueError(
                    "`num_time_bins` must be provided when `temporal_mode != 'none'`."
                )
            if num_time_bins <= 0:
                raise ValueError(
                    f"`num_time_bins` must be > 0, got {num_time_bins}."
                )

        if temporal_hidden_dim <= 0:
            raise ValueError(
                f"`temporal_hidden_dim` must be > 0, got {temporal_hidden_dim}."
            )

        if num_lstm_layers <= 0:
            raise ValueError(
                f"`num_lstm_layers` must be > 0, got {num_lstm_layers}."
            )

        if graph_pool not in {"sum", "mean", "max"}:
            raise ValueError(
                f"`graph_pool` must be one of ('sum', 'mean', 'max'), got {graph_pool!r}."
            )

        if head_hidden_dim <= 0:
            raise ValueError(
                f"`head_hidden_dim` must be > 0, got {head_hidden_dim}."
            )

    def get_encoder_modules(self) -> Iterable[nn.Module]:
        """
        Return encoder-side modules for transfer learning.

        Returns:
            Iterable of modules considered part of the transferable encoder.
        """
        modules = [self.preprocess, self.gat_encoder]

        if self.temporal_mode == "fc":
            modules.append(self.temporal_mlp)
        elif self.temporal_mode == "lstm":
            modules.extend(
                [
                    self.temporal_input_norm,
                    self.temporal_lstm,
                    self.temporal_output_norm,
                ]
            )

        return modules

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
            2. temporal encoder (if any)
            3. prediction head
        """
        super().reset_parameters()

        if self.temporal_mode == "fc":
            for module in self.temporal_mlp:
                _reset_module_parameters(module)
        elif self.temporal_mode == "lstm":
            _reset_module_parameters(self.temporal_input_norm)
            _reset_module_parameters(self.temporal_lstm)
            _reset_module_parameters(self.temporal_output_norm)

        _reset_module_parameters(self.head)

    def forward(self, data: Data) -> torch.Tensor:
        """
        Run the full forward pass.

        Required fields:
            Always required:
                - data.x: [num_nodes, input_dim]
                - data.edge_index: [2, num_edges]
                - data.batch: [num_nodes]

            Additionally required if `temporal_mode != "none"`:
                - data.time_bin: [num_nodes], integer bin id for each node

        Shape contract:
            - node encoder output: [num_nodes, embedding_dim]
            - final output: [batch_size, output_dim]

        Args:
            data:
                Batched PyG Data object.

        Returns:
            Prediction tensor of shape [batch_size, output_dim].

        Raises:
            ValueError:
                If required fields are missing or inconsistent.
        """
        self.validate_data(data, require_batch=True)

        x = self.encode(data.x, data.edge_index)

        if self.temporal_mode == "none":
            graph_repr = self._graph_pool(x, data.batch)
        else:
            self._validate_temporal_data(data)
            pooled = self._pool_by_time_bins(
                x=x,
                time_bin=data.time_bin,
                batch=data.batch,
                num_time_bins=self.num_time_bins,
            )

            if self.temporal_mode == "fc":
                pooled_flat = pooled.reshape(pooled.size(0), -1)
                graph_repr = self.temporal_mlp(pooled_flat)
            else:
                graph_repr = self._encode_temporal_sequence(pooled)

        out = self.head(graph_repr)
        return out

    def _validate_temporal_data(self, data: Data) -> None:
        """
        Validate temporal metadata required for time-bin aggregation.

        Required:
            - `data.time_bin` exists
            - `data.time_bin` is a 1D LongTensor of shape [num_nodes]
            - values lie in [0, num_time_bins - 1]

        Args:
            data:
                Input batched graph data.

        Raises:
            ValueError:
                If time-bin metadata is missing or malformed.
            TypeError:
                If time-bin tensor type is invalid.
        """
        if not hasattr(data, "time_bin") or data.time_bin is None:
            raise ValueError(
                "`data.time_bin` is required when `temporal_mode != 'none'`."
            )

        if not torch.is_tensor(data.time_bin):
            raise TypeError("`data.time_bin` must be a torch.Tensor.")

        if data.time_bin.dtype != torch.long:
            raise TypeError(
                f"`data.time_bin` must have dtype torch.long, got {data.time_bin.dtype}."
            )

        if data.time_bin.dim() != 1:
            raise ValueError(
                f"`data.time_bin` must be 1D [num_nodes], got shape {tuple(data.time_bin.shape)}."
            )

        if data.time_bin.size(0) != data.x.size(0):
            raise ValueError(
                "`data.time_bin` length must match number of nodes. "
                f"Got {data.time_bin.size(0)} and {data.x.size(0)}."
            )

        min_bin = int(data.time_bin.min().item())
        max_bin = int(data.time_bin.max().item())

        if min_bin < 0 or max_bin >= self.num_time_bins:
            raise ValueError(
                "`data.time_bin` contains values outside configured range "
                f"[0, {self.num_time_bins - 1}]. Found min={min_bin}, max={max_bin}."
            )

    def _graph_pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Apply graph-level pooling to node embeddings.

        Args:
            x:
                Node embeddings [num_nodes, embedding_dim].
            batch:
                Batch vector [num_nodes].

        Returns:
            Graph embeddings [batch_size, embedding_dim].
        """
        if self.graph_pool == "sum":
            return global_add_pool(x, batch)
        if self.graph_pool == "mean":
            return global_mean_pool(x, batch)
        if self.graph_pool == "max":
            return global_max_pool(x, batch)

        raise RuntimeError(f"Unsupported graph_pool={self.graph_pool!r}.")

    @staticmethod
    def _pool_by_time_bins(
        x: torch.Tensor,
        time_bin: torch.Tensor,
        batch: torch.Tensor,
        num_time_bins: int,
    ) -> torch.Tensor:
        """
        Aggregate node embeddings by (graph_id, time_bin).

        Aggregation rule:
            For each graph in the batch and each time bin, node embeddings are
            summed. Missing bins are filled with zeros.

        Additional feature:
            The scalar time-bin index is concatenated as the last feature
            dimension for each bin.

        Shape invariants:
            - x.shape[0] == time_bin.shape[0] == batch.shape[0]
            - output.shape == [batch_size, num_time_bins, embedding_dim + 1]

        Args:
            x:
                Node embeddings [num_nodes, embedding_dim].
            time_bin:
                Time-bin labels [num_nodes].
            batch:
                Graph batch vector [num_nodes].
            num_time_bins:
                Total number of bins.

        Returns:
            Time-binned graph representation
            [batch_size, num_time_bins, embedding_dim + 1].
        """
        batch_size = int(batch.max().item()) + 1
        embedding_dim = x.size(1)

        group_index = batch * num_time_bins + time_bin
        pooled = scatter(
            x,
            group_index,
            dim=0,
            dim_size=batch_size * num_time_bins,
            reduce="sum",
        )
        pooled = pooled.view(batch_size, num_time_bins, embedding_dim)

        time_values = torch.arange(
            num_time_bins, device=x.device, dtype=x.dtype
        ).view(1, num_time_bins, 1)
        time_values = time_values.expand(batch_size, num_time_bins, 1)

        pooled = torch.cat([pooled, time_values], dim=-1)
        return pooled

    def _encode_temporal_sequence(self, pooled: torch.Tensor) -> torch.Tensor:
        """
        Encode time-binned sequences with a bidirectional LSTM.

        Processing steps:
            1. LayerNorm on input sequence
            2. BiLSTM over time bins
            3. LayerNorm on recurrent outputs
            4. Dropout
            5. Mean pooling over the temporal dimension

        Args:
            pooled:
                Time-binned tensor [batch_size, num_time_bins, embedding_dim + 1].

        Returns:
            Graph representation [batch_size, temporal_hidden_dim * 2].
        """
        pooled = self.temporal_input_norm(pooled)
        lstm_out, _ = self.temporal_lstm(pooled)
        lstm_out = self.temporal_output_norm(lstm_out)
        lstm_out = self.dropout(lstm_out)
        return lstm_out.mean(dim=1)

    def extra_repr(self) -> str:
        """
        Extra representation shown in `print(model)`.
        """
        return (
            f"output_dim={self.output_dim}, "
            f"temporal_mode={self.temporal_mode}, "
            f"num_time_bins={self.num_time_bins}, "
            f"temporal_hidden_dim={self.temporal_hidden_dim}, "
            f"num_lstm_layers={self.num_lstm_layers}, "
            f"graph_pool={self.graph_pool}, "
            f"head_hidden_dim={self.head_hidden_dim}, "
            f"output_positive={self.output_positive}"
        )
