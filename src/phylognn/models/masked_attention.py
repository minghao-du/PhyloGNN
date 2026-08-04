"""Masked-attention regression constrained by a leaf Laplacian."""

from __future__ import annotations

import math

import torch
from torch import nn


class MaskedAttentionPhyloRegressor(nn.Module):
    """Regress one value per leaf from padded position representations.

    Args:
        input_dim: Width of each position representation.
        hidden_dim: Width of the projected leaf representation.
        leaf_laplacian: Normalized leaf Laplacian with shape ``[N, N]``.

    The forward method accepts representations with shape ``[N, L, input_dim]``
    and a position mask with shape ``[N, L]``. It returns predictions ``[N]``
    and attention ``[N, L]`` whose masked positions are exactly zero.
    """

    def __init__(self, input_dim: int, hidden_dim: int, leaf_laplacian: torch.Tensor) -> None:
        super().__init__()
        if isinstance(input_dim, bool) or not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError("`input_dim` must be a positive integer.")
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError("`hidden_dim` must be a positive integer.")
        if not torch.is_tensor(leaf_laplacian):
            raise TypeError("`leaf_laplacian` must be a torch.Tensor.")
        if (
            leaf_laplacian.ndim != 2
            or leaf_laplacian.size(0) == 0
            or leaf_laplacian.size(0) != leaf_laplacian.size(1)
        ):
            raise ValueError(
                "`leaf_laplacian` must have nonempty square shape [N, N]; "
                f"got {tuple(leaf_laplacian.shape)}."
            )
        if leaf_laplacian.dtype != torch.float32:
            raise ValueError("`leaf_laplacian` must have dtype torch.float32.")
        if not torch.isfinite(leaf_laplacian).all():
            raise ValueError("`leaf_laplacian` must contain only finite values.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.position_projection = nn.Linear(input_dim, hidden_dim)
        self.attention_scorer = nn.Linear(hidden_dim, 1)
        self.regression_head = nn.Linear(hidden_dim, 1)
        self.raw_alpha = nn.Parameter(torch.tensor(math.log(0.1 / 0.9)))
        self.register_buffer("leaf_laplacian", leaf_laplacian.detach().clone())

    def forward(
        self, representations: torch.Tensor, position_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one prediction and an attention distribution for each leaf."""
        if not torch.is_tensor(representations):
            raise TypeError("`representations` must be a torch.Tensor.")
        if not torch.is_tensor(position_mask):
            raise TypeError("`position_mask` must be a torch.Tensor.")
        if representations.ndim != 3:
            raise ValueError(
                "`representations` must have shape [N, L, D]; "
                f"got {tuple(representations.shape)}."
            )
        if representations.size(-1) != self.input_dim:
            raise ValueError("`representations` last dimension must match `input_dim`.")
        if representations.dtype != torch.float32:
            raise ValueError("`representations` must have dtype torch.float32.")
        if not torch.isfinite(representations).all():
            raise ValueError("`representations` must contain only finite values.")
        if position_mask.shape != representations.shape[:2]:
            raise ValueError(
                "`position_mask` must have shape [N, L] matching `representations`; "
                f"got {tuple(position_mask.shape)}."
            )
        if representations.size(0) != self.leaf_laplacian.size(0):
            raise ValueError(
                "`representations` leaf count must match `leaf_laplacian`; "
                f"got {representations.size(0)} and {self.leaf_laplacian.size(0)}."
            )

        mask = position_mask.to(dtype=torch.bool)
        if not torch.all(mask.any(dim=1)):
            raise ValueError("Every `position_mask` row must contain a valid position.")

        projected = torch.tanh(self.position_projection(representations))
        scores = self.attention_scorer(projected).squeeze(-1)
        attention = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=1)
        attention = attention * mask.to(dtype=attention.dtype)
        pooled = (attention.unsqueeze(-1) * projected).sum(dim=1)
        smoothed = pooled - torch.sigmoid(self.raw_alpha) * (self.leaf_laplacian @ pooled)
        return self.regression_head(smoothed).squeeze(-1), attention
