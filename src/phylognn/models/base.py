"""
Base model abstractions for phylogenetic GNN architectures.

This module defines abstract base classes and shared utilities for building
phylogenetic tree analysis models with PyTorch and PyTorch Geometric.

Design goals:
    1. Provide a consistent interface for all phylogenetic GNN models.
    2. Centralize common logic such as parameter counting, freezing, resetting,
       and graph input validation.
    3. Make encoder reuse easy for transfer learning and downstream tasks.

Conventions:
    - All graph inputs are expected to be `torch_geometric.data.Data`.
    - Node-level encoders should expose `encode(...)` and `get_embedding_dim()`.
    - Graph-level or task-specific prediction is implemented in subclasses.

Notes:
    - Freezing behavior is based on explicit encoder/head module boundaries,
      not fragile parameter-name heuristics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Literal, Optional

import torch
import torch.nn as nn
from torch_geometric.data import Data

from .layers import build_gat_encoder, validate_positive_int_counters

GATEncoderType = Literal["gat", "res_gat"]


def _reset_module_parameters(module: nn.Module) -> None:
    """
    Reset parameters of a module if it exposes `reset_parameters()`.

    Args:
        module:
            A PyTorch module instance.

    Notes:
        - This function is safe for modules that do not implement
          `reset_parameters()`: in that case it is a no-op.
    """
    if hasattr(module, "reset_parameters") and callable(module.reset_parameters):
        module.reset_parameters()


class BasePhyloGNN(nn.Module, ABC):
    """
    Abstract base class for phylogenetic GNN models.

    Subclasses must implement:
        - forward(data): task-specific forward pass
        - get_embedding_dim(): encoder output dimension

    Common utilities provided:
        - graph input validation
        - parameter counting
        - encoder freezing / full unfreezing
        - module parameter resetting

    Engineering notes:
        - This base class does not rely on parameter-name heuristics to
          separate encoder and head modules.
        - Subclasses should override `get_encoder_modules()` and
          `get_head_modules()` when they contain task-specific heads.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, data: Data) -> torch.Tensor:
        """
        Run the model forward pass.

        Args:
            data:
                PyTorch Geometric Data object.

        Returns:
            Model predictions.
        """
        raise NotImplementedError

    @abstractmethod
    def get_embedding_dim(self) -> int:
        """
        Return the output dimension of learned node embeddings.

        Returns:
            Encoder output dimension.
        """
        raise NotImplementedError

    def validate_data(self, data: Data, require_batch: bool = False) -> None:
        """
        Validate a PyG Data object for graph neural network processing.

        Requirements:
            - `data` is an instance of `torch_geometric.data.Data`
            - `data.x` exists and is a 2D tensor: [num_nodes, num_features]
            - `data.edge_index` exists and is a LongTensor: [2, num_edges]
            - if `require_batch=True`, `data.batch` exists and is a LongTensor
              of shape [num_nodes]

        Additional consistency checks:
            - number of nodes must be positive
            - edge indices must be within [0, num_nodes - 1]
            - if batch is required, `len(batch)` must equal number of nodes

        Args:
            data:
                Input graph object.
            require_batch:
                Whether graph batch assignment is required.

        Raises:
            TypeError:
                If object types or tensor dtypes are invalid.
            ValueError:
                If required fields are missing, malformed, or inconsistent.
        """
        if not isinstance(data, Data):
            raise TypeError(
                "Expected `data` to be an instance of "
                f"`torch_geometric.data.Data`, got `{type(data).__name__}`."
            )

        if not hasattr(data, "x") or data.x is None:
            raise ValueError("Input `data` must contain node features `x`.")
        if not torch.is_tensor(data.x):
            raise TypeError("`data.x` must be a torch.Tensor.")
        if data.x.dim() != 2:
            raise ValueError(
                "`data.x` must be a 2D tensor of shape "
                f"[num_nodes, num_features], got shape {tuple(data.x.shape)}."
            )
        if not data.x.is_floating_point():
            raise TypeError("`data.x` must use a floating-point dtype.")
        if data.x.size(0) <= 0:
            raise ValueError("`data.x` must contain at least one node.")

        if not hasattr(data, "edge_index") or data.edge_index is None:
            raise ValueError("Input `data` must contain graph connectivity `edge_index`.")
        if not torch.is_tensor(data.edge_index):
            raise TypeError("`data.edge_index` must be a torch.Tensor.")
        if data.edge_index.dtype != torch.long:
            raise TypeError(
                f"`data.edge_index` must have dtype torch.long, got {data.edge_index.dtype}."
            )
        if data.edge_index.dim() != 2 or data.edge_index.size(0) != 2:
            raise ValueError(
                "`data.edge_index` must have shape [2, num_edges], "
                f"got shape {tuple(data.edge_index.shape)}."
            )

        self._validate_edge_index_bounds(data.edge_index, num_nodes=data.x.size(0))

        if require_batch:
            if not hasattr(data, "batch") or data.batch is None:
                raise ValueError("Input `data` must contain `batch` for batched graph operations.")
            if not torch.is_tensor(data.batch):
                raise TypeError("`data.batch` must be a torch.Tensor.")
            if data.batch.dtype != torch.long:
                raise TypeError(f"`data.batch` must have dtype torch.long, got {data.batch.dtype}.")
            if data.batch.dim() != 1:
                raise ValueError(
                    f"`data.batch` must be 1D of shape [num_nodes], got {tuple(data.batch.shape)}."
                )
            if data.batch.size(0) != data.x.size(0):
                raise ValueError(
                    "`data.batch` length must match number of nodes. "
                    f"Got len(batch)={data.batch.size(0)} and num_nodes={data.x.size(0)}."
                )

    @staticmethod
    def _validate_edge_index_bounds(edge_index: torch.Tensor, num_nodes: int) -> None:
        """
        Validate that `edge_index` references only valid node ids.

        Args:
            edge_index:
                Edge index tensor of shape [2, num_edges].
            num_nodes:
                Number of nodes in the graph.

        Raises:
            ValueError:
                If edge indices contain negative or out-of-range node ids.
        """
        if edge_index.numel() == 0:
            return

        min_idx = int(edge_index.min().item())
        max_idx = int(edge_index.max().item())

        if min_idx < 0 or max_idx >= num_nodes:
            raise ValueError(
                "`data.edge_index` contains node ids outside valid range "
                f"[0, {num_nodes - 1}]. Found min={min_idx}, max={max_idx}."
            )

    def get_num_parameters(self, trainable_only: bool = True) -> int:
        """
        Count model parameters.

        Args:
            trainable_only:
                If True, count only parameters with `requires_grad=True`.

        Returns:
            Parameter count.
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def get_encoder_modules(self) -> Iterable[nn.Module]:
        """
        Return modules that belong to the transferable encoder.

        Subclasses should override this method if they expose encoder modules.

        Returns:
            Iterable of encoder modules.
        """
        return []

    def get_head_modules(self) -> Iterable[nn.Module]:
        """
        Return modules that belong to the task-specific prediction head.

        Subclasses should override this method if they expose prediction heads.

        Returns:
            Iterable of head modules.
        """
        return []

    def freeze_encoder(self) -> None:
        """
        Freeze encoder parameters while keeping head parameters trainable.

        Behavior:
            - All parameters in `get_encoder_modules()` are frozen.
            - All parameters in `get_head_modules()` remain trainable.

        Notes:
            - Any parameters not included in either collection are left unchanged.
            - Subclasses should avoid returning overlapping modules.
        """
        for module in self.get_encoder_modules():
            for param in module.parameters():
                param.requires_grad = False

        for module in self.get_head_modules():
            for param in module.parameters():
                param.requires_grad = True

    def unfreeze_all(self) -> None:
        """
        Make all model parameters trainable.
        """
        for param in self.parameters():
            param.requires_grad = True

    def reset_parameters(self) -> None:
        """
        Reset parameters of immediate child modules that expose `reset_parameters()`.

        Notes:
            - This method only resets direct child modules.
            - Nested reset behavior should be handled by each child module's own
              `reset_parameters()` implementation.
        """
        for module in self.children():
            _reset_module_parameters(module)


class BaseGATNet(BasePhyloGNN):
    """
    Base GAT encoder with optional feature preprocessing and pluggable GAT backbone.

    Architecture:
        input node features
            -> optional preprocessing projection
            -> dropout
            -> GAT encoder stack (plain or residual)
            -> dropout
            -> node embeddings

    This class provides reusable node-level encoding only. Subclasses are
    responsible for graph-level pooling and task-specific prediction heads.

    Args:
        input_dim:
            Input node feature dimension.
        preprocess_dim:
            Output dimension of the optional preprocessing layer.
            Required if `use_preprocessing=True`.
        gat_hidden_dim:
            Hidden dimension per attention head in each GAT layer.
        gat_heads:
            Number of attention heads.
        num_gat_layers:
            Number of GAT blocks in the encoder stack. Must be a positive
            non-bool Python integer.
        dropout_prob:
            Dropout probability used in the encoder pipeline.
        use_preprocessing:
            Whether to apply a learnable preprocessing projection.
        encoder_type:
            Type of GAT encoder backbone:
                - "gat": plain stacked GAT
                - "res_gat": residual stacked GAT

    Output embedding dimension:
        `gat_hidden_dim * gat_heads`
    """

    def __init__(
        self,
        input_dim: int,
        preprocess_dim: Optional[int],
        gat_hidden_dim: int,
        gat_heads: int = 4,
        num_gat_layers: int = 3,
        dropout_prob: float = 0.2,
        use_preprocessing: bool = True,
        encoder_type: GATEncoderType = "res_gat",
    ) -> None:
        super().__init__()

        self._validate_init_args(
            input_dim=input_dim,
            preprocess_dim=preprocess_dim,
            gat_hidden_dim=gat_hidden_dim,
            gat_heads=gat_heads,
            num_gat_layers=num_gat_layers,
            dropout_prob=dropout_prob,
            use_preprocessing=use_preprocessing,
            encoder_type=encoder_type,
        )

        self.input_dim = input_dim
        self.preprocess_dim = preprocess_dim
        self.gat_hidden_dim = gat_hidden_dim
        self.gat_heads = gat_heads
        self.num_gat_layers = num_gat_layers
        self.dropout_prob = dropout_prob
        self.use_preprocessing = use_preprocessing
        self.encoder_type = encoder_type

        if self.use_preprocessing:
            self.preprocess = nn.Sequential(
                nn.Linear(input_dim, preprocess_dim),
                nn.ReLU(),
            )
            gat_input_dim = preprocess_dim
        else:
            self.preprocess = nn.Identity()
            gat_input_dim = input_dim

        self.gat_encoder = build_gat_encoder(
            encoder_type=encoder_type,
            in_channels=gat_input_dim,
            hidden_channels=gat_hidden_dim,
            num_layers=num_gat_layers,
            heads=gat_heads,
            dropout_prob=dropout_prob,
        )

        self.dropout = nn.Dropout(dropout_prob)

    @staticmethod
    def _validate_init_args(
        input_dim: int,
        preprocess_dim: Optional[int],
        gat_hidden_dim: int,
        gat_heads: int,
        num_gat_layers: int,
        dropout_prob: float,
        use_preprocessing: bool,
        encoder_type: GATEncoderType,
    ) -> None:
        """
        Validate encoder constructor arguments.

        Raises:
            ValueError:
                If any argument is invalid.
        """
        if input_dim <= 0:
            raise ValueError(f"`input_dim` must be > 0, got {input_dim}.")
        if gat_hidden_dim <= 0:
            raise ValueError(f"`gat_hidden_dim` must be > 0, got {gat_hidden_dim}.")
        if gat_heads <= 0:
            raise ValueError(f"`gat_heads` must be > 0, got {gat_heads}.")
        validate_positive_int_counters(num_gat_layers=num_gat_layers)
        if not (0.0 <= dropout_prob < 1.0):
            raise ValueError(f"`dropout_prob` must be in [0, 1), got {dropout_prob}.")
        if use_preprocessing:
            if preprocess_dim is None:
                raise ValueError("`preprocess_dim` must be provided when `use_preprocessing=True`.")
            if preprocess_dim <= 0:
                raise ValueError(f"`preprocess_dim` must be > 0 when used, got {preprocess_dim}.")
        if encoder_type not in {"gat", "res_gat"}:
            raise ValueError(
                f"`encoder_type` must be one of ('gat', 'res_gat'), got {encoder_type!r}."
            )

    def get_encoder_modules(self) -> Iterable[nn.Module]:
        """
        Return transferable encoder modules.

        Returns:
            Encoder module collection.
        """
        return [self.preprocess, self.gat_encoder]

    def reset_parameters(self) -> None:
        """
        Reset preprocessing and GAT encoder parameters.
        """
        _reset_module_parameters(self.preprocess)
        _reset_module_parameters(self.gat_encoder)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Encode node features into node-level embeddings.

        Contract:
            - `x`: [num_nodes, input_dim]
            - `edge_index`: [2, num_edges]
            - output: [num_nodes, gat_hidden_dim * gat_heads]

        Notes:
            - This method performs node-level encoding only.
            - No graph-level pooling or prediction head is applied here.
            - Dropout behavior follows standard PyTorch train/eval semantics.

        Args:
            x:
                Node feature matrix.
            edge_index:
                Graph connectivity in COO format.

        Returns:
            Encoded node embeddings.
        """
        x = self.preprocess(x)
        x = self.dropout(x)
        x = self.gat_encoder(x, edge_index)
        x = self.dropout(x)
        return x

    def encode_data(self, data: Data) -> torch.Tensor:
        """
        Convenience wrapper that encodes directly from a PyG Data object.

        Required fields:
            - `data.x`
            - `data.edge_index`

        Args:
            data:
                Input graph data.

        Returns:
            Node embeddings of shape [num_nodes, embedding_dim].
        """
        self.validate_data(data, require_batch=False)
        return self.encode(data.x, data.edge_index)

    def get_embedding_dim(self) -> int:
        """
        Return encoder output embedding dimension.

        Returns:
            `gat_hidden_dim * gat_heads`
        """
        return self.gat_hidden_dim * self.gat_heads

    def extra_repr(self) -> str:
        """
        Extra representation shown in `print(model)`.
        """
        return (
            f"input_dim={self.input_dim}, "
            f"preprocess_dim={self.preprocess_dim}, "
            f"gat_hidden_dim={self.gat_hidden_dim}, "
            f"gat_heads={self.gat_heads}, "
            f"num_gat_layers={self.num_gat_layers}, "
            f"dropout_prob={self.dropout_prob}, "
            f"use_preprocessing={self.use_preprocessing}, "
            f"encoder_type={self.encoder_type}"
        )

    @abstractmethod
    def forward(self, data: Data) -> torch.Tensor:
        """
        Subclass-defined task-specific forward pass.

        Args:
            data:
                PyG Data object.

        Returns:
            Task-specific output tensor.
        """
        raise NotImplementedError
