"""Sparse-query sequence regression with a phylogenetic residual branch."""

from __future__ import annotations

import math
from numbers import Real
from typing import Literal, Sequence

import torch
from entmax import entmax15
from torch import nn

_SUPPORTED_ATTENTION_NORMALIZATIONS = ("softmax", "entmax15")


def _validate_positive_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"`{name}` must be a positive integer.")


def _validate_nonnegative_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"`{name}` must be a non-negative integer.")


def _validate_dropout(value: object, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) < 1.0
    ):
        raise ValueError(f"`{name}` must be a finite number in [0, 1).")


def _validate_kernel_sizes(kernel_sizes: object) -> tuple[int, ...]:
    if isinstance(kernel_sizes, (str, bytes)) or not isinstance(kernel_sizes, Sequence):
        raise TypeError("`cnn_kernel_sizes` must be a nonempty sequence of positive odd integers.")
    resolved = tuple(kernel_sizes)
    if not resolved or any(
        isinstance(size, bool) or not isinstance(size, int) or size <= 0 or size % 2 == 0
        for size in resolved
    ):
        raise ValueError("`cnn_kernel_sizes` must contain only positive odd integers.")
    return resolved


class _MaskedMultiscaleResidualBlock(nn.Module):
    """Mix masked token features across several local receptive fields."""

    def __init__(self, token_dim: int, kernel_sizes: tuple[int, ...], dropout_prob: float) -> None:
        super().__init__()
        self.depthwise_convolutions = nn.ModuleList(
            [
                nn.Conv1d(
                    token_dim,
                    token_dim,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    groups=token_dim,
                )
                for kernel_size in kernel_sizes
            ]
        )
        self.pointwise_mixing = nn.Conv1d(token_dim * len(kernel_sizes), token_dim, kernel_size=1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_prob)
        self.normalization = nn.LayerNorm(token_dim)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return locally mixed tokens with invalid positions kept at zero."""
        mask_values = mask.unsqueeze(-1).to(dtype=tokens.dtype)
        masked_tokens = tokens * mask_values
        channels_first = masked_tokens.transpose(1, 2)
        multiscale = torch.cat(
            [convolution(channels_first) for convolution in self.depthwise_convolutions],
            dim=1,
        )
        mixed = self.pointwise_mixing(multiscale).transpose(1, 2)
        output = self.normalization(masked_tokens + self.dropout(self.activation(mixed)))
        return output * mask_values


class SparseQueryPhyloRegressor(nn.Module):
    """Regress one value per leaf from padded position representations.

    The sequence encoder can run on arbitrary mini-batches through
    :meth:`encode_sequences`. After all species embeddings have been assembled
    in leaf-Laplacian order, :meth:`predict_from_embeddings` applies the
    sequence and phylogenetic prediction branches. :meth:`forward` combines
    both stages and therefore requires all leaves at once.

    Args:
        input_dim: Width of each input position representation.
        leaf_laplacian: Fixed leaf Laplacian with shape ``[N, N]``.
        adapter_rank: Low-rank adapter bottleneck width.
        token_dim: Position-level hidden width.
        num_cnn_blocks: Number of multiscale residual CNN blocks; zero disables
            local mixing.
        cnn_kernel_sizes: Positive odd receptive-field widths for each block.
        num_queries: Number of independent phenotype attention queries.
        slot_dim: Key, value, and attention-slot width for every query.
        species_dim: Width of each pooled species embedding.
        sequence_hidden_dim: Hidden width of the sequence prediction head.
        phylogeny_hidden_dim: Hidden width of the phylogenetic residual head.
        adapter_dropout_prob: Dropout probability inside the input adapter.
        cnn_dropout_prob: Dropout probability in every residual CNN block.
        representation_dropout_prob: Dropout probability in the species
            representation head.
        sequence_dropout_prob: Dropout probability in the sequence head.
        phylogeny_dropout_prob: Dropout probability in the phylogenetic head.
        attention_normalization: ``"entmax15"`` for sparse attention or
            ``"softmax"`` for the dense ablation.
        phylogeny_gate_init: Initial phylogenetic residual gate in ``(0, 1)``.

    Position masks must be nonempty right-padded rows. Boolean masks and numeric
    masks containing exactly zero and one are accepted. Attention has shape
    ``[B, num_queries, L]`` and is exactly zero at padding positions.
    """

    def __init__(
        self,
        input_dim: int,
        leaf_laplacian: torch.Tensor,
        adapter_rank: int = 32,
        token_dim: int = 64,
        num_cnn_blocks: int = 1,
        cnn_kernel_sizes: Sequence[int] = (3, 9, 27),
        num_queries: int = 4,
        slot_dim: int = 32,
        species_dim: int = 64,
        sequence_hidden_dim: int = 32,
        phylogeny_hidden_dim: int = 32,
        adapter_dropout_prob: float = 0.0,
        cnn_dropout_prob: float = 0.3,
        representation_dropout_prob: float = 0.3,
        sequence_dropout_prob: float = 0.3,
        phylogeny_dropout_prob: float = 0.3,
        attention_normalization: Literal["softmax", "entmax15"] = "entmax15",
        phylogeny_gate_init: float = 0.05,
    ) -> None:
        super().__init__()
        for name, value in (
            ("input_dim", input_dim),
            ("adapter_rank", adapter_rank),
            ("token_dim", token_dim),
            ("num_queries", num_queries),
            ("slot_dim", slot_dim),
            ("species_dim", species_dim),
            ("sequence_hidden_dim", sequence_hidden_dim),
            ("phylogeny_hidden_dim", phylogeny_hidden_dim),
        ):
            _validate_positive_integer(value, name)
        _validate_nonnegative_integer(num_cnn_blocks, "num_cnn_blocks")
        resolved_kernel_sizes = _validate_kernel_sizes(cnn_kernel_sizes)
        for name, value in (
            ("adapter_dropout_prob", adapter_dropout_prob),
            ("cnn_dropout_prob", cnn_dropout_prob),
            ("representation_dropout_prob", representation_dropout_prob),
            ("sequence_dropout_prob", sequence_dropout_prob),
            ("phylogeny_dropout_prob", phylogeny_dropout_prob),
        ):
            _validate_dropout(value, name)
        if not isinstance(attention_normalization, str):
            raise TypeError("`attention_normalization` must be a string.")
        if attention_normalization not in _SUPPORTED_ATTENTION_NORMALIZATIONS:
            supported = ", ".join(repr(value) for value in _SUPPORTED_ATTENTION_NORMALIZATIONS)
            raise ValueError(
                f"`attention_normalization` must be one of {supported}; "
                f"got {attention_normalization!r}."
            )
        if (
            isinstance(phylogeny_gate_init, bool)
            or not isinstance(phylogeny_gate_init, Real)
            or not math.isfinite(float(phylogeny_gate_init))
            or not 0.0 < float(phylogeny_gate_init) < 1.0
        ):
            raise ValueError("`phylogeny_gate_init` must be a finite number in (0, 1).")
        self._validate_laplacian(leaf_laplacian)

        self.input_dim = input_dim
        self.adapter_rank = adapter_rank
        self.token_dim = token_dim
        self.num_cnn_blocks = num_cnn_blocks
        self.cnn_kernel_sizes = resolved_kernel_sizes
        self.num_queries = num_queries
        self.slot_dim = slot_dim
        self.species_dim = species_dim
        self.sequence_hidden_dim = sequence_hidden_dim
        self.phylogeny_hidden_dim = phylogeny_hidden_dim
        self.attention_normalization: str = attention_normalization

        self.adapter = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, adapter_rank),
            nn.GELU(),
            nn.Dropout(adapter_dropout_prob),
            nn.Linear(adapter_rank, token_dim),
            nn.LayerNorm(token_dim),
        )
        self.cnn_blocks = nn.ModuleList(
            [
                _MaskedMultiscaleResidualBlock(token_dim, resolved_kernel_sizes, cnn_dropout_prob)
                for _ in range(num_cnn_blocks)
            ]
        )
        self.key_projections = nn.ModuleList(
            [nn.Linear(token_dim, slot_dim) for _ in range(num_queries)]
        )
        self.value_projections = nn.ModuleList(
            [nn.Linear(token_dim, slot_dim) for _ in range(num_queries)]
        )
        self.query_vectors = nn.ParameterList(
            [nn.Parameter(torch.empty(slot_dim)) for _ in range(num_queries)]
        )
        for query_vector in self.query_vectors:
            nn.init.normal_(query_vector, std=slot_dim**-0.5)

        pooled_dim = num_queries * slot_dim + 2 * token_dim
        self.representation_head = nn.Sequential(
            nn.Linear(pooled_dim, species_dim),
            nn.GELU(),
            nn.Dropout(representation_dropout_prob),
            nn.LayerNorm(species_dim),
        )
        self.sequence_head = nn.Sequential(
            nn.LayerNorm(species_dim),
            nn.Linear(species_dim, sequence_hidden_dim),
            nn.GELU(),
            nn.Dropout(sequence_dropout_prob),
            nn.Linear(sequence_hidden_dim, 1),
        )
        self.phylogeny_first_order = nn.Linear(species_dim, phylogeny_hidden_dim, bias=True)
        self.phylogeny_second_order = nn.Linear(species_dim, phylogeny_hidden_dim, bias=False)
        self.phylogeny_activation = nn.GELU()
        self.phylogeny_dropout = nn.Dropout(phylogeny_dropout_prob)
        self.phylogeny_output = nn.Linear(phylogeny_hidden_dim, 1)
        self.raw_beta = nn.Parameter(
            torch.tensor(math.log(float(phylogeny_gate_init) / (1.0 - phylogeny_gate_init)))
        )
        self.register_buffer("leaf_laplacian", leaf_laplacian.detach().clone())

    @staticmethod
    def _validate_laplacian(leaf_laplacian: object) -> None:
        if not torch.is_tensor(leaf_laplacian):
            raise TypeError("`leaf_laplacian` must be a torch.Tensor.")
        if (
            leaf_laplacian.ndim != 2
            or leaf_laplacian.size(0) == 0
            or leaf_laplacian.size(0) != leaf_laplacian.size(1)
        ):
            raise ValueError("`leaf_laplacian` must have nonempty square shape [N, N].")
        if not leaf_laplacian.is_floating_point():
            raise ValueError("`leaf_laplacian` must have a floating-point dtype.")
        if not torch.isfinite(leaf_laplacian).all():
            raise ValueError("`leaf_laplacian` must contain only finite values.")

    @staticmethod
    def _resolve_position_mask(
        position_mask: torch.Tensor, expected_shape: torch.Size
    ) -> torch.Tensor:
        if not torch.is_tensor(position_mask):
            raise TypeError("`position_mask` must be a torch.Tensor.")
        if position_mask.shape != expected_shape:
            raise ValueError(
                "`position_mask` must have shape [B, L] matching `representations`; "
                f"got {tuple(position_mask.shape)}."
            )
        if position_mask.dtype != torch.bool:
            if position_mask.is_complex() or not torch.all(
                (position_mask == 0) | (position_mask == 1)
            ):
                raise ValueError("`position_mask` must contain only zero and one.")
        mask = position_mask.to(dtype=torch.bool)
        if not torch.all(mask.any(dim=1)):
            raise ValueError("Every `position_mask` row must contain a valid position.")
        if mask.size(1) > 1 and torch.any((~mask[:, :-1]) & mask[:, 1:]):
            raise ValueError("`position_mask` rows must use contiguous right padding.")
        return mask

    def _normalize_attention(self, scores: torch.Tensor) -> torch.Tensor:
        if self.attention_normalization == "entmax15":
            return entmax15(scores, dim=-1)
        return torch.softmax(scores, dim=-1)

    def encode_sequences(
        self, representations: torch.Tensor, position_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a sequence mini-batch into species embeddings and attention."""
        if not torch.is_tensor(representations):
            raise TypeError("`representations` must be a torch.Tensor.")
        if representations.ndim != 3:
            raise ValueError(
                "`representations` must have shape [B, L, D]; "
                f"got {tuple(representations.shape)}."
            )
        if representations.size(-1) != self.input_dim:
            raise ValueError("`representations` last dimension must match `input_dim`.")
        if not representations.is_floating_point():
            raise ValueError("`representations` must have a floating-point dtype.")
        if not torch.isfinite(representations).all():
            raise ValueError("`representations` must contain only finite values.")
        mask = self._resolve_position_mask(position_mask, representations.shape[:2])
        if mask.device != representations.device:
            raise ValueError("`position_mask` must be on the same device as `representations`.")

        mask_values = mask.unsqueeze(-1).to(dtype=representations.dtype)
        tokens = self.adapter(representations) * mask_values
        for block in self.cnn_blocks:
            tokens = block(tokens, mask)

        attentions = []
        slots = []
        score_scale = math.sqrt(self.slot_dim)
        for key_projection, value_projection, query_vector in zip(
            self.key_projections, self.value_projections, self.query_vectors
        ):
            keys = key_projection(tokens)
            values = value_projection(tokens)
            scores = torch.einsum("bld,d->bl", keys, query_vector) / score_scale
            scores = scores.masked_fill(~mask, float("-inf"))
            attention = self._normalize_attention(scores)
            attention = attention * mask.to(dtype=attention.dtype)
            attentions.append(attention)
            slots.append(torch.sum(attention.unsqueeze(-1) * values, dim=1))

        attention_tensor = torch.stack(attentions, dim=1)
        valid_counts = mask_values.sum(dim=1)
        masked_mean = tokens.sum(dim=1) / valid_counts
        masked_max = tokens.masked_fill(~mask.unsqueeze(-1), float("-inf")).amax(dim=1)
        pooled = torch.cat([*slots, masked_mean, masked_max], dim=-1)
        return self.representation_head(pooled), attention_tensor

    def predict_from_embeddings(
        self, species_embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return final, sequence, and phylogenetic predictions for all leaves."""
        if not torch.is_tensor(species_embeddings):
            raise TypeError("`species_embeddings` must be a torch.Tensor.")
        if species_embeddings.ndim != 2 or species_embeddings.size(-1) != self.species_dim:
            raise ValueError(
                "`species_embeddings` must have shape [N, species_dim]; "
                f"got {tuple(species_embeddings.shape)}."
            )
        if species_embeddings.size(0) != self.leaf_laplacian.size(0):
            raise ValueError(
                "`species_embeddings` leaf count must match `leaf_laplacian`; "
                f"got {species_embeddings.size(0)} and {self.leaf_laplacian.size(0)}."
            )
        if not species_embeddings.is_floating_point():
            raise ValueError("`species_embeddings` must have a floating-point dtype.")
        if not torch.isfinite(species_embeddings).all():
            raise ValueError("`species_embeddings` must contain only finite values.")
        if (
            species_embeddings.device != self.leaf_laplacian.device
            or species_embeddings.dtype != self.leaf_laplacian.dtype
        ):
            raise ValueError(
                "`species_embeddings` and `leaf_laplacian` must have the same dtype and device."
            )

        sequence_prediction = self.sequence_head(species_embeddings).squeeze(-1)
        first_order = species_embeddings - self.leaf_laplacian @ species_embeddings
        second_order = first_order - self.leaf_laplacian @ first_order
        phylogeny_hidden = self.phylogeny_activation(
            self.phylogeny_first_order(first_order) + self.phylogeny_second_order(second_order)
        )
        phylogeny_prediction = self.phylogeny_output(
            self.phylogeny_dropout(phylogeny_hidden)
        ).squeeze(-1)
        prediction = sequence_prediction + torch.sigmoid(self.raw_beta) * phylogeny_prediction
        return prediction, sequence_prediction, phylogeny_prediction

    def forward(
        self, representations: torch.Tensor, position_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one prediction and per-query attention for every leaf."""
        species_embeddings, attention = self.encode_sequences(representations, position_mask)
        prediction = self.predict_from_embeddings(species_embeddings)[0]
        return prediction, attention
