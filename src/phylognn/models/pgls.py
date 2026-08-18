"""PGLS regression projection head."""

from __future__ import annotations

import torch
from torch import nn


class PGLSRegressionHead(nn.Module):
    """Project ordered leaf representations to one or more trait predictions.

    Parameters
    ----------
    input_dim : int
        Width of each input representation ``D``; must be positive.
    output_dim : int
        Number of predicted traits ``T``; must be positive.

    Inputs are floating-point tensors of shape ``[N, D]`` and the output has
    shape ``[N, T]``. Input dtype and device must match the trainable linear
    parameters; no implicit conversion is performed.

    Returns
    -------
    torch.Tensor
        Ordered trait predictions of shape ``[N, T]``, preserving the input
        dtype and device.

    Raises
    ------
    TypeError
        If dimensions or the input object type are invalid.
    ValueError
        If dimensions, rank, width, dtype, device, or finiteness are invalid.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        if isinstance(input_dim, bool) or not isinstance(input_dim, int):
            raise TypeError("input_dim must be a positive integer.")
        if isinstance(output_dim, bool) or not isinstance(output_dim, int):
            raise TypeError("output_dim must be a positive integer.")
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive.")
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, representations: torch.Tensor) -> torch.Tensor:
        """Return trait predictions for ``representations`` of shape ``[N, D]``.

        Raises ``TypeError`` for an invalid object and ``ValueError`` for an
        invalid rank, batch size, width, dtype, device, or non-finite value.
        """
        if not torch.is_tensor(representations):
            raise TypeError("representations must be a torch.Tensor.")
        if representations.ndim != 2:
            raise ValueError("representations must have shape [N, D].")
        if representations.shape[0] == 0:
            raise ValueError("representations must contain at least one row.")
        if representations.shape[1] != self.input_dim:
            raise ValueError(
                f"representations width must be {self.input_dim}, got {representations.shape[1]}."
            )
        if representations.dtype not in (torch.float32, torch.float64):
            raise ValueError("representations dtype must be torch.float32 or torch.float64.")
        if representations.dtype != self.linear.weight.dtype:
            raise ValueError("representations dtype must match head parameter dtype.")
        if representations.device != self.linear.weight.device:
            raise ValueError("representations device must match head parameter device.")
        if not torch.isfinite(representations).all():
            raise ValueError("representations must contain only finite values.")
        return self.linear(representations)
