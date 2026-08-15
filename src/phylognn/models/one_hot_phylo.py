"""One-hot sequence regression with relative bins and a phylogenetic branch."""

from __future__ import annotations

import math
from numbers import Real

import torch
from torch import nn


class OneHotPhyloRegressor(nn.Module):
    """Regress one phenotype value per leaf from mRNA one-hot arrays.

    The expected input has shape ``[N, L, 7]`` with the channels
    ``A, C, G, T, N, CDS, non_CDS``.  Valid positions are described by a
    contiguous-right-padded mask ``[N, L]``.  Each sequence is divided into
    ``num_bins`` equal relative-coordinate bins before a small convolutional
    encoder is applied.  This makes the representation insensitive to
    species-specific indels and to the common padding length.

    A global composition/length branch is combined with the binned sequence
    branch.  The resulting species embeddings are passed through first- and
    second-order leaf-Laplacian features, and the phylogenetic prediction is
    added through a small learnable gate initialized near zero.

    Args:
        input_dim: Number of channels.  One-hot mRNA input requires ``7``.
        leaf_laplacian: Float32 leaf Laplacian with shape ``[N, N]``.
        num_bins: Number of relative-coordinate bins.
        hidden_dim: Width of the binned and global encoders.
        species_dim: Width of the per-leaf embedding.
        phylogeny_hidden_dim: Width of the phylogenetic prediction branch.
        dropout_prob: Dropout probability used in the encoders and heads.
        phylogeny_gate_init: Initial contribution of the phylogenetic branch.

    The model returns only a prediction tensor with shape ``[N]``.  The
    standard leaf-regression fitting API accepts prediction-only models, so no
    artificial position attention is reported for the binned representation.
    """

    expected_input_dim = 7

    def __init__(
        self,
        input_dim: int,
        leaf_laplacian: torch.Tensor,
        num_bins: int = 64,
        hidden_dim: int = 64,
        species_dim: int = 64,
        phylogeny_hidden_dim: int = 32,
        dropout_prob: float = 0.1,
        phylogeny_gate_init: float = 0.05,
    ) -> None:
        super().__init__()
        self._validate_integer(input_dim, "input_dim")
        if input_dim != self.expected_input_dim:
            raise ValueError(
                f"`input_dim` must be {self.expected_input_dim} for mRNA one-hot input; "
                f"got {input_dim}."
            )
        for name, value in (
            ("num_bins", num_bins),
            ("hidden_dim", hidden_dim),
            ("species_dim", species_dim),
            ("phylogeny_hidden_dim", phylogeny_hidden_dim),
        ):
            self._validate_integer(value, name)
        if (
            isinstance(dropout_prob, bool)
            or not isinstance(dropout_prob, Real)
            or not math.isfinite(float(dropout_prob))
            or not 0.0 <= float(dropout_prob) < 1.0
        ):
            raise ValueError("`dropout_prob` must be a finite number in [0, 1).")
        if (
            isinstance(phylogeny_gate_init, bool)
            or not isinstance(phylogeny_gate_init, Real)
            or not math.isfinite(float(phylogeny_gate_init))
            or not 0.0 < float(phylogeny_gate_init) < 1.0
        ):
            raise ValueError("`phylogeny_gate_init` must be a finite number in (0, 1).")
        self._validate_laplacian(leaf_laplacian)

        self.input_dim = input_dim
        self.num_bins = num_bins
        self.hidden_dim = hidden_dim
        self.species_dim = species_dim
        self.phylogeny_hidden_dim = phylogeny_hidden_dim
        self.dropout_prob = float(dropout_prob)

        self.bin_encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout_prob),
        )
        self.bin_normalization = nn.LayerNorm(hidden_dim)
        self.global_encoder = nn.Sequential(
            nn.Linear(input_dim + 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_prob),
        )
        self.species_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 3, species_dim),
            nn.GELU(),
            nn.LayerNorm(species_dim),
            nn.Dropout(dropout_prob),
        )
        self.sequence_head = nn.Sequential(
            nn.Linear(species_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, 1),
        )
        self.phylogeny_head = nn.Sequential(
            nn.Linear(species_dim * 3, phylogeny_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(phylogeny_hidden_dim, 1),
        )
        self.raw_phylogeny_gate = nn.Parameter(
            torch.tensor(math.log(float(phylogeny_gate_init) / (1.0 - phylogeny_gate_init)))
        )
        self.register_buffer("leaf_laplacian", leaf_laplacian.detach().clone())

    @staticmethod
    def _validate_integer(value: object, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"`{name}` must be a positive integer.")

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
        if leaf_laplacian.dtype != torch.float32:
            raise ValueError("`leaf_laplacian` must have dtype torch.float32.")
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
                "`position_mask` must have shape [N, L] matching `representations`; "
                f"got {tuple(position_mask.shape)}."
            )
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
        return mask

    def encode_sequences(
        self, representations: torch.Tensor, position_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode one relative-binned species embedding per leaf."""
        if not torch.is_tensor(representations):
            raise TypeError("`representations` must be a torch.Tensor.")
        if representations.ndim != 3:
            raise ValueError(
                "`representations` must have shape [N, L, 7]; "
                f"got {tuple(representations.shape)}."
            )
        if representations.size(-1) != self.input_dim:
            raise ValueError("`representations` last dimension must match `input_dim`.")
        if representations.size(0) == 0 or representations.size(1) == 0:
            raise ValueError("`representations` must have nonempty shape [N, L, 7].")
        if representations.dtype != torch.float32:
            raise ValueError("`representations` must have dtype torch.float32.")
        if representations.device != self.leaf_laplacian.device:
            raise ValueError("`representations` must be on the same device as `leaf_laplacian`.")
        if representations.size(0) != self.leaf_laplacian.size(0):
            raise ValueError(
                "`representations` leaf count must match `leaf_laplacian`; "
                f"got {representations.size(0)} and {self.leaf_laplacian.size(0)}."
            )
        if position_mask.device != representations.device:
            raise ValueError("`position_mask` must be on the same device as `representations`.")
        if not torch.isfinite(representations).all():
            raise ValueError("`representations` must contain only finite values.")

        mask = self._resolve_position_mask(position_mask, representations.shape[:2])
        lengths = mask.sum(dim=1).to(dtype=torch.long)
        batch_size, sequence_length, _ = representations.shape
        positions = torch.arange(sequence_length, device=representations.device)
        bin_indices = (
            positions.unsqueeze(0) * self.num_bins // lengths.unsqueeze(1)
        ).clamp(max=self.num_bins - 1)

        mask_values = mask.unsqueeze(-1).to(dtype=representations.dtype)
        valid_values = representations * mask_values
        bin_sums = torch.zeros(
            batch_size,
            self.num_bins,
            self.input_dim,
            dtype=representations.dtype,
            device=representations.device,
        )
        bin_sums.scatter_add_(
            1,
            bin_indices.unsqueeze(-1).expand(-1, -1, self.input_dim),
            valid_values,
        )
        bin_counts = torch.zeros(
            batch_size,
            self.num_bins,
            1,
            dtype=representations.dtype,
            device=representations.device,
        )
        bin_counts.scatter_add_(
            1,
            bin_indices.unsqueeze(-1),
            mask_values,
        )
        bin_valid = bin_counts.squeeze(-1) > 0
        binned = bin_sums / bin_counts.clamp_min(1.0)
        binned = binned * bin_valid.unsqueeze(-1).to(dtype=representations.dtype)

        encoded_bins = self.bin_encoder(binned.transpose(1, 2)).transpose(1, 2)
        encoded_bins = self.bin_normalization(encoded_bins)
        encoded_bins = encoded_bins * bin_valid.unsqueeze(-1).to(dtype=encoded_bins.dtype)
        valid_bin_counts = bin_valid.sum(dim=1, keepdim=True).to(dtype=encoded_bins.dtype)
        binned_mean = encoded_bins.sum(dim=1) / valid_bin_counts.clamp_min(1.0)
        binned_max = encoded_bins.masked_fill(~bin_valid.unsqueeze(-1), float("-inf")).amax(dim=1)

        composition = valid_values.sum(dim=1) / lengths.unsqueeze(-1).to(
            dtype=representations.dtype
        )
        length_values = lengths.to(dtype=representations.dtype)
        length_reference = length_values.max().clamp_min(1.0)
        normalized_length = length_values / length_reference
        log_length = torch.log1p(length_values) / torch.log1p(length_reference)
        global_features = torch.cat(
            [composition, normalized_length.unsqueeze(-1), log_length.unsqueeze(-1)], dim=-1
        )
        global_embedding = self.global_encoder(global_features)
        return self.species_encoder(torch.cat([binned_mean, binned_max, global_embedding], dim=-1))

    def predict_from_embeddings(self, species_embeddings: torch.Tensor) -> torch.Tensor:
        """Predict from precomputed species embeddings in leaf order."""
        if species_embeddings.ndim != 2 or species_embeddings.size(-1) != self.species_dim:
            raise ValueError(
                "`species_embeddings` must have shape [N, species_dim]; "
                f"got {tuple(species_embeddings.shape)}."
            )
        if species_embeddings.size(0) != self.leaf_laplacian.size(0):
            raise ValueError("`species_embeddings` leaf count must match `leaf_laplacian`.")
        if species_embeddings.dtype != self.leaf_laplacian.dtype:
            raise ValueError("`species_embeddings` must have dtype torch.float32.")
        if species_embeddings.device != self.leaf_laplacian.device:
            raise ValueError("`species_embeddings` must be on the same device as `leaf_laplacian`.")
        if not torch.isfinite(species_embeddings).all():
            raise ValueError("`species_embeddings` must contain only finite values.")

        first_order = species_embeddings - self.leaf_laplacian @ species_embeddings
        second_order = first_order - self.leaf_laplacian @ first_order
        phylogeny_features = torch.cat([species_embeddings, first_order, second_order], dim=-1)
        sequence_prediction = self.sequence_head(species_embeddings).squeeze(-1)
        phylogeny_prediction = self.phylogeny_head(phylogeny_features).squeeze(-1)
        return sequence_prediction + torch.sigmoid(self.raw_phylogeny_gate) * phylogeny_prediction

    def forward(self, representations: torch.Tensor, position_mask: torch.Tensor) -> torch.Tensor:
        """Return one phenotype prediction for each leaf."""
        species_embeddings = self.encode_sequences(representations, position_mask)
        return self.predict_from_embeddings(species_embeddings)
