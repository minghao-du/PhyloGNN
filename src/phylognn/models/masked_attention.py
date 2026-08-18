"""Masked-attention regression constrained by a leaf Laplacian."""

from __future__ import annotations

import math
from typing import Literal

import torch
from entmax import entmax15
from torch import nn

_SUPPORTED_ATTENTION_NORMALIZATIONS = ("softmax", "entmax15")


class MaskedAttentionPhyloRegressor(nn.Module):
    """Regress one value per leaf from padded position representations.

    Args:
        input_dim: Width of each position representation.
        hidden_dim: Width of the projected leaf representation.
        leaf_laplacian: Normalized leaf Laplacian with shape ``[N, N]``.
        dropout_prob: Dropout probability applied to projected representations.
        attention_normalization: Row-wise normalization applied to masked
            attention scores. ``"softmax"`` (the default) produces a dense
            distribution; ``"entmax15"`` uses the 1.5-entmax transform from the
            ``entmax`` package and may assign exact zero weight to low-scoring
            valid positions. The selected string is available as the
            ``attention_normalization`` attribute.
        chunk_size: Optional positive, non-boolean integer limiting the number
            of leaves processed by the raw position projection per invocation.
            ``None`` preserves full-batch raw encoding.

    The forward method accepts representations with shape ``[N, L, input_dim]``
    and a position mask with shape ``[N, L]``. It returns predictions ``[N]``
    and attention ``[N, L]``. The canonical raw position encoder is
    ``position_projection``; chunk outputs are concatenated in input leaf order,
    then attention, pooling, regression, and Laplacian work run once over the
    complete batch. Numeric binary masks are normalized to boolean on their
    existing device, and representation finiteness is checked per consumed
    chunk. With dropout disabled, full and chunked outputs compare with
    ``torch.allclose(..., atol=1e-6, rtol=0)``. Attention is finite,
    non-negative, row-normalized within ``1e-6``, and exactly zero at masked
    positions; every mask row must contain at least one valid position.
    Construction raises ``TypeError`` for non-string ``attention_normalization``
    values and ``ValueError`` for strings other than ``"softmax"`` and
    ``"entmax15"``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        leaf_laplacian: torch.Tensor,
        dropout_prob: float = 0.3,
        attention_normalization: Literal["softmax", "entmax15"] = "softmax",
        chunk_size: int | None = None,
    ) -> None:
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
        if not (0.0 <= dropout_prob < 1.0):
            raise ValueError(f"`dropout_prob` must be in [0, 1), got {dropout_prob}.")

        if not isinstance(attention_normalization, str):
            raise TypeError(
                "`attention_normalization` must be a string, "
                f"got {type(attention_normalization).__name__}. Accepted values: "
                f"{', '.join(repr(v) for v in _SUPPORTED_ATTENTION_NORMALIZATIONS)}."
            )
        if attention_normalization not in _SUPPORTED_ATTENTION_NORMALIZATIONS:
            raise ValueError(
                "`attention_normalization` must be one of "
                f"{', '.join(repr(v) for v in _SUPPORTED_ATTENTION_NORMALIZATIONS)}; "
                f"got {attention_normalization!r}."
            )
        if chunk_size is not None and (
            isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0
        ):
            raise ValueError("`chunk_size` must be a positive integer.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob
        self.attention_normalization: str = attention_normalization
        self.chunk_size = chunk_size
        self.position_projection = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.attention_scorer = nn.Linear(hidden_dim, 1)
        self.regression_head = nn.Linear(hidden_dim, 1)
        self.raw_alpha = nn.Parameter(torch.tensor(math.log(0.1 / 0.9)))
        self.register_buffer("leaf_laplacian", leaf_laplacian.detach().clone())

    def _forward_leaf_features(
        self, representations: torch.Tensor, position_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ordered pre-prediction features and attention for each leaf.

        Global shape, dtype, device, and mask contracts are validated before
        raw position encoding. The downstream attention and graph computation
        remains a single full-batch pass after any chunked projection.
        """
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
        if representations.size(0) == 0 or representations.size(1) == 0:
            raise ValueError("`representations` must have nonempty shape [N, L, D].")
        if representations.dtype != torch.float32:
            raise ValueError("`representations` must have dtype torch.float32.")
        if position_mask.shape != representations.shape[:2]:
            raise ValueError(
                "`position_mask` must have shape [N, L] matching `representations`; "
                f"got {tuple(position_mask.shape)}."
            )
        if position_mask.device != representations.device:
            raise ValueError("`position_mask` must be on the same device as `representations`.")
        if representations.size(0) != self.leaf_laplacian.size(0):
            raise ValueError(
                "`representations` leaf count must match `leaf_laplacian`; "
                f"got {representations.size(0)} and {self.leaf_laplacian.size(0)}."
            )
        if representations.device != self.leaf_laplacian.device:
            raise ValueError("`representations` must be on the same device as `leaf_laplacian`.")

        if position_mask.dtype != torch.bool:
            if position_mask.is_complex():
                raise TypeError("`position_mask` must contain boolean or real numeric values.")
            if not torch.isfinite(position_mask).all():
                raise ValueError("`position_mask` must contain only finite values.")
            if not torch.all((position_mask == 0) | (position_mask == 1)):
                raise ValueError("`position_mask` must contain only zero and one.")
        mask = position_mask.to(dtype=torch.bool)
        if not torch.all(mask.any(dim=1)):
            raise ValueError("Every `position_mask` row must contain a valid position.")
        if mask.size(1) > 1 and torch.any((~mask[:, :-1]) & mask[:, 1:]):
            raise ValueError("`position_mask` rows must use contiguous right padding.")

        chunk_size = self.chunk_size or representations.size(0)
        projected_chunks = []
        for start in range(0, representations.size(0), chunk_size):
            chunk = representations[start : start + chunk_size]
            if not torch.isfinite(chunk).all():
                raise ValueError("`representations` must contain only finite values.")
            projected_chunks.append(self.dropout(torch.tanh(self.position_projection(chunk))))
        projected = torch.cat(projected_chunks, dim=0)
        scores = self.attention_scorer(projected).squeeze(-1)
        masked_scores = scores.masked_fill(~mask, float("-inf"))

        if self.attention_normalization == "entmax15":
            attention = entmax15(masked_scores, dim=1)
        else:
            attention = torch.softmax(masked_scores, dim=1)

        attention = attention * mask.to(dtype=attention.dtype)
        pooled = (attention.unsqueeze(-1) * projected).sum(dim=1)
        smoothed = pooled - torch.sigmoid(self.raw_alpha) * (self.leaf_laplacian @ pooled)
        return smoothed, attention

    def forward_leaf_representations(
        self, representations: torch.Tensor, position_mask: torch.Tensor
    ) -> torch.Tensor:
        """Return ordered final leaf features with shape ``[N, hidden_dim]``.

        Args:
            representations: Validated float32 position features with shape
                ``[N, L, input_dim]``.
            position_mask: Boolean valid-position mask with shape ``[N, L]``.

        Returns:
            Nonempty ordered features immediately before scalar prediction.

        Raises:
            TypeError: If either input is not a compatible tensor.
            ValueError: If shape, dtype, device, mask, or finiteness contracts fail.
        """
        features, _ = self._forward_leaf_features(representations, position_mask)
        return features

    def forward(
        self, representations: torch.Tensor, position_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one prediction and attention distribution for each leaf."""
        features, attention = self._forward_leaf_features(representations, position_mask)
        return self.regression_head(features).squeeze(-1), attention
