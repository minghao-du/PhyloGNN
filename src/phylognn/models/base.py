"""
Base model classes for phylogenetic GNN architectures.

This module defines abstract base classes and shared utilities for building
phylogenetic tree analysis models with PyTorch and PyTorch Geometric.

Design goals:
    1. Provide a consistent interface for all phylogenetic GNN models.
    2. Centralize common logic such as parameter counting, freezing, and
       graph input validation.
    3. Make encoder-only reuse easy for transfer learning and downstream tasks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.data import Data

from .layers import ResidualGATStack


def _reset_module_parameters(module: nn.Module) -> None:
    """
    Safely reset parameters of a module if it implements `reset_parameters`.

    Args:
        module: PyTorch module.
    """
    if hasattr(module, "reset_parameters") and callable(module.reset_parameters):
        module.reset_parameters()


class BasePhyloGNN(nn.Module, ABC):
    """
    Abstract base class for phylogenetic GNN models.

    This class defines the common interface and shared utility methods used by
    all phylogenetic GNN architectures in the package.

    Subclasses must implement:
        - forward(data): task-specific forward pass
        - get_embedding_dim(): output embedding dimension of the encoder

    Notes:
        - The input is expected to be a PyTorch Geometric `Data` object.
        - By default, this base class assumes a model may contain an encoder
          and a task-specific prediction head.
        - The default `freeze_encoder()` behavior uses parameter-name heuristics.
          Subclasses can override `_is_head_parameter()` for stricter control.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            data:
                PyTorch Geometric Data object. Expected fields depend on the
                concrete model, but typically include:
                    - x: Node feature matrix [num_nodes, num_features]
                    - edge_index: Graph connectivity [2, num_edges]
                    - batch: Batch vector [num_nodes] (optional for some tasks)

        Returns:
            Model predictions.
        """
        raise NotImplementedError

    @abstractmethod
    def get_embedding_dim(self) -> int:
        """
        Return the dimension of learned node embeddings.

        Returns:
            Embedding dimension as an integer.
        """
        raise NotImplementedError

    def validate_data(self, data: Data, require_batch: bool = False) -> None:
        """
        Validate that a PyG Data object contains the required attributes.

        Args:
            data: Input graph data.
            require_batch: If True, also require `data.batch`.

        Raises:
            TypeError: If `data` is not a `torch_geometric.data.Data` instance.
            ValueError: If required fields are missing or malformed.
        """
        if not isinstance(data, Data):
            raise TypeError(
                f"Expected `data` to be an instance of torch_geometric.data.Data, "
                f"but got {type(data).__name__}."
            )

        if not hasattr(data, "x") or data.x is None:
            raise ValueError("Input `data` must contain node features `x`.")

        if not hasattr(data, "edge_index") or data.edge_index is None:
            raise ValueError("Input `data` must contain graph connectivity `edge_index`.")

        if data.x.dim() != 2:
            raise ValueError(
                f"`data.x` must be a 2D tensor of shape [num_nodes, num_features], "
                f"but got shape {tuple(data.x.shape)}."
            )

        if data.edge_index.dim() != 2 or data.edge_index.size(0) != 2:
            raise ValueError(
                f"`data.edge_index` must have shape [2, num_edges], "
                f"but got shape {tuple(data.edge_index.shape)}."
            )

        if require_batch:
            if not hasattr(data, "batch") or data.batch is None:
                raise ValueError(
                    "Input `data` must contain `batch` for batched graph operations."
                )

    def get_num_parameters(self, trainable_only: bool = True) -> int:
        """
        Count model parameters.

        Args:
            trainable_only: If True, count only parameters with
                `requires_grad=True`. Otherwise count all parameters.

        Returns:
            Number of parameters.
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def _is_head_parameter(self, name: str) -> bool:
        """
        Heuristic to identify whether a parameter belongs to a prediction head.

        Subclasses may override this method if they use a different naming
        convention for task-specific modules.

        Args:
            name: Parameter name from `named_parameters()`.

        Returns:
            True if the parameter is considered part of a prediction head.
        """
        head_keywords = ("head", "fc", "classifier", "predictor", "readout", "mlp")
        return any(keyword in name.lower() for keyword in head_keywords)

    def freeze_encoder(self) -> None:
        """
        Freeze encoder parameters for transfer learning.

        By default, parameters identified as belonging to the prediction head
        remain trainable, while all other parameters are frozen.

        Notes:
            - This method relies on `_is_head_parameter()`.
            - If your subclass uses a custom head naming scheme, override
              `_is_head_parameter()` for precise behavior.
        """
        for name, param in self.named_parameters():
            param.requires_grad = self._is_head_parameter(name)

    def unfreeze_all(self) -> None:
        """
        Unfreeze all model parameters.
        """
        for param in self.parameters():
            param.requires_grad = True

    def reset_parameters(self) -> None:
        """
        Reset parameters of all child modules that implement `reset_parameters`.

        This is useful for repeated experiments, cross-validation, and
        hyperparameter search.
        """
        for module in self.children():
            _reset_module_parameters(module)


class BaseGATNet(BasePhyloGNN):
    """
    Base GAT network with configurable architecture.

    Architecture:
        input features
            -> optional preprocessing layer
            -> residual GAT encoder stack
            -> optional dropout on encoded embeddings
            -> task-specific head (implemented by subclasses)

    Args:
        input_dim: Dimension of input node features.
        preprocess_dim:
            Output dimension of the preprocessing layer.
            Required when `use_preprocessing=True`.
        gat_hidden_dim: Hidden dimension per GAT head.
        gat_heads: Number of attention heads.
        num_gat_layers: Number of GAT layers in the encoder.
        dropout_prob: Dropout probability applied in the base encoder pipeline.
        use_preprocessing: Whether to apply a linear preprocessing layer before GAT.

    Output embedding dimension:
        `gat_hidden_dim * gat_heads`

    Example:
        >>> class MyGATModel(BaseGATNet):
        ...     def __init__(self, input_dim, num_classes):
        ...         super().__init__(
        ...             input_dim=input_dim,
        ...             preprocess_dim=64,
        ...             gat_hidden_dim=32,
        ...             gat_heads=4,
        ...             num_gat_layers=3,
        ...             dropout_prob=0.2,
        ...             use_preprocessing=True,
        ...         )
        ...         self.head = nn.Linear(self.get_embedding_dim(), num_classes)
        ...
        ...     def forward(self, data):
        ...         self.validate_data(data, require_batch=False)
        ...         x = self.encode_data(data)
        ...         return self.head(x)
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
        )

        self.input_dim = input_dim
        self.preprocess_dim = preprocess_dim
        self.gat_hidden_dim = gat_hidden_dim
        self.gat_heads = gat_heads
        self.num_gat_layers = num_gat_layers
        self.dropout_prob = dropout_prob
        self.use_preprocessing = use_preprocessing

        # Preprocessing layer
        if self.use_preprocessing:
            self.preprocess = nn.Sequential(
                nn.Linear(input_dim, preprocess_dim),
                nn.ReLU(),
            )
            gat_input_dim = preprocess_dim
        else:
            self.preprocess = nn.Identity()
            gat_input_dim = input_dim

        # Residual GAT encoder
        self.gat_encoder = ResidualGATStack(
            in_channels=gat_input_dim,
            hidden_channels=gat_hidden_dim,
            num_layers=num_gat_layers,
            heads=gat_heads,
            dropout_prob=dropout_prob,
        )

        # Shared dropout for encoder pipeline
        self.dropout = nn.Dropout(dropout_prob)

        # Ensure all resettable submodules are initialized explicitly
        self.reset_parameters()

    @staticmethod
    def _validate_init_args(
        input_dim: int,
        preprocess_dim: Optional[int],
        gat_hidden_dim: int,
        gat_heads: int,
        num_gat_layers: int,
        dropout_prob: float,
        use_preprocessing: bool,
    ) -> None:
        """
        Validate constructor arguments.

        Raises:
            ValueError: If any argument is invalid.
        """
        if input_dim <= 0:
            raise ValueError(f"`input_dim` must be > 0, got {input_dim}.")
        if gat_hidden_dim <= 0:
            raise ValueError(f"`gat_hidden_dim` must be > 0, got {gat_hidden_dim}.")
        if gat_heads <= 0:
            raise ValueError(f"`gat_heads` must be > 0, got {gat_heads}.")
        if num_gat_layers <= 0:
            raise ValueError(f"`num_gat_layers` must be > 0, got {num_gat_layers}.")
        if not (0.0 <= dropout_prob < 1.0):
            raise ValueError(
                f"`dropout_prob` must be in [0, 1), got {dropout_prob}."
            )
        if use_preprocessing:
            if preprocess_dim is None:
                raise ValueError(
                    "`preprocess_dim` must be provided when `use_preprocessing=True`."
                )
            if preprocess_dim <= 0:
                raise ValueError(
                    f"`preprocess_dim` must be > 0 when used, got {preprocess_dim}."
                )

    def reset_parameters(self) -> None:
        """
        Reset parameters of preprocessing and GAT encoder modules.
        """
        _reset_module_parameters(self.preprocess)
        _reset_module_parameters(self.gat_encoder)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Encode node features using preprocessing and GAT layers.

        Args:
            x: Node feature matrix of shape [num_nodes, input_dim].
            edge_index: Graph connectivity tensor of shape [2, num_edges].

        Returns:
            Encoded node features of shape
            [num_nodes, gat_hidden_dim * gat_heads].
        """
        x = self.preprocess(x)
        x = self.dropout(x)
        x = self.gat_encoder(x, edge_index)
        x = self.dropout(x)
        return x

    def encode_data(self, data: Data) -> torch.Tensor:
        """
        Convenience wrapper to encode directly from a PyG Data object.

        Args:
            data: PyTorch Geometric Data object containing at least
                `x` and `edge_index`.

        Returns:
            Encoded node embeddings.
        """
        self.validate_data(data, require_batch=False)
        return self.encode(data.x, data.edge_index)

    def get_embedding_dim(self) -> int:
        """
        Return the output embedding dimension of the GAT encoder.

        Returns:
            Embedding dimension = `gat_hidden_dim * gat_heads`.
        """
        return self.gat_hidden_dim * self.gat_heads

    def extra_repr(self) -> str:
        """
        Extra string representation shown in `print(model)`.
        """
        return (
            f"input_dim={self.input_dim}, "
            f"preprocess_dim={self.preprocess_dim}, "
            f"gat_hidden_dim={self.gat_hidden_dim}, "
            f"gat_heads={self.gat_heads}, "
            f"num_gat_layers={self.num_gat_layers}, "
            f"dropout_prob={self.dropout_prob}, "
            f"use_preprocessing={self.use_preprocessing}"
        )

    @abstractmethod
    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass to be implemented by subclasses.

        Args:
            data: PyTorch Geometric Data object.

        Returns:
            Task-specific model output.
        """
        raise NotImplementedError
