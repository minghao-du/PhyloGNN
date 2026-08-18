"""Deterministic fixtures for the PGLS regression head and loss tests."""

import torch
import pytest

from phylognn.models import PGLSRegressionHead
from phylognn.training import PGLSLoss


def test_pgls_head_and_loss_are_curated_public_exports():
    import phylognn.models as models
    import phylognn.training as training

    assert models.PGLSRegressionHead is PGLSRegressionHead
    assert "PGLSRegressionHead" in models.__all__
    assert training.PGLSLoss is PGLSLoss
    assert "PGLSLoss" in training.__all__


def test_head_preserves_shape_order_and_dtype():
    head = PGLSRegressionHead(2, 2).double()
    with torch.no_grad():
        head.linear.weight.copy_(torch.eye(2))
        head.linear.bias.zero_()
    representations = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    assert torch.equal(head(representations), representations)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_head_supports_matching_float_dtypes(dtype):
    head = PGLSRegressionHead(3, 1).to(dtype=dtype)
    output = head(torch.ones(4, 3, dtype=dtype))
    assert output.shape == (4, 1)
    assert output.dtype == dtype
    assert output.device == head.linear.weight.device


def test_head_rejects_width_and_rank_mismatch():
    head = PGLSRegressionHead(2, 1)
    with pytest.raises(ValueError):
        head(torch.ones(3, 3))
    with pytest.raises(ValueError):
        head(torch.ones(3, 2, 1))


def test_loss_single_trait_normalizes_targets_and_one_leaf():
    predictions, targets, covariances, batch = _single_trait_pgls_inputs()
    loss = PGLSLoss()(predictions, targets, covariances, batch)
    expected = PGLSLoss()(predictions, targets.unsqueeze(1), covariances, batch)
    assert torch.allclose(loss, expected)
    one = _one_leaf_pgls_inputs()
    assert torch.isfinite(PGLSLoss()(*one))


def test_loss_matches_reference_for_multi_tree_multi_trait_and_has_gradients():
    predictions, targets, covariances, batch = _multi_tree_pgls_inputs()
    predictions = torch.cat((predictions, predictions + 0.25), dim=1).requires_grad_()
    targets = torch.cat((targets, targets - 0.1), dim=1)
    loss = PGLSLoss()(predictions, targets, covariances, batch)
    expected_terms = []
    for tree_id, covariance in enumerate(covariances):
        idx = batch == tree_id
        residual = targets[idx] - predictions[idx]
        expected_terms.append(
            (residual * torch.linalg.solve(covariance, residual)).sum(0).mean() / idx.sum()
        )
    assert torch.allclose(loss, torch.stack(expected_terms).mean())
    loss.backward()
    assert predictions.grad is not None and torch.isfinite(predictions.grad).all()


def test_loss_uses_one_solve_per_tree(monkeypatch):
    predictions, targets, covariances, batch = _multi_tree_pgls_inputs()
    predictions = torch.cat((predictions, predictions + 0.25), dim=1)
    targets = torch.cat((targets, targets - 0.1), dim=1)
    right_hand_side_shapes = []
    original = torch.linalg.solve

    def counted(*args, **kwargs):
        right_hand_side_shapes.append(args[1].shape)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.linalg, "solve", counted)
    PGLSLoss()(predictions, targets, covariances, batch)
    assert right_hand_side_shapes == [torch.Size((2, 2)), torch.Size((3, 2))]


@pytest.mark.parametrize(
    ("representations", "error_type", "message"),
    [
        (None, TypeError, "torch.Tensor"),
        (torch.empty(0, 2), ValueError, "at least one row"),
        (torch.ones(2), ValueError, r"shape \[N, D\]"),
        (torch.ones(2, 2, 1), ValueError, r"shape \[N, D\]"),
        (torch.ones(2, 3), ValueError, "width"),
        (torch.ones(2, 2, dtype=torch.int64), ValueError, "float32 or torch.float64"),
        (torch.full((2, 2), torch.nan), ValueError, "finite"),
        (torch.full((2, 2), torch.inf), ValueError, "finite"),
    ],
)
def test_head_rejects_malformed_representations(representations, error_type, message):
    head = PGLSRegressionHead(2, 1)
    with pytest.raises(error_type, match=message):
        head(representations)


def test_head_rejects_parameter_dtype_and_device_mismatches():
    head = PGLSRegressionHead(2, 1).double()
    with pytest.raises(ValueError, match="dtype.*head parameter"):
        head(torch.ones(2, 2, dtype=torch.float32))

    cpu_head = PGLSRegressionHead(2, 1)
    with pytest.raises(ValueError, match="device.*head parameter"):
        cpu_head(torch.ones(2, 2, device="meta"))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("predictions", None, "predictions.*torch.Tensor"),
        ("targets", None, "targets.*torch.Tensor"),
        ("batch", None, "batch.*torch.Tensor"),
        ("covariances", (torch.eye(3, dtype=torch.float64),), "covariances.*list"),
        ("covariances", [None], "covariance.*torch.Tensor"),
    ],
)
def test_loss_rejects_wrong_object_types(field, value, message):
    predictions, targets, covariances, batch = _single_tree_pgls_inputs()
    arguments = {
        "predictions": predictions,
        "targets": targets,
        "covariances": covariances,
        "batch": batch,
    }
    arguments[field] = value
    with pytest.raises(TypeError, match=message):
        PGLSLoss()(**arguments)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("predictions", torch.ones(3, dtype=torch.float64), r"predictions.*\[N, T\]"),
        ("targets", torch.ones(3, 1, 1, dtype=torch.float64), r"targets.*\[N, T\]"),
        ("predictions", torch.empty(0, 1, dtype=torch.float64), "non-empty"),
        ("predictions", torch.empty(3, 0, dtype=torch.float64), "non-empty"),
        ("targets", torch.ones(2, 1, dtype=torch.float64), "matching shapes"),
        ("batch", torch.zeros((3, 1), dtype=torch.long), r"batch.*\[N\]"),
        ("batch", torch.zeros(2, dtype=torch.long), r"batch.*\[N\]"),
        ("covariances", [], "non-empty list"),
    ],
)
def test_loss_rejects_empty_rank_and_shape_mismatches(field, value, message):
    predictions, targets, covariances, batch = _single_tree_pgls_inputs()
    arguments = {
        "predictions": predictions,
        "targets": targets,
        "covariances": covariances,
        "batch": batch,
    }
    arguments[field] = value
    with pytest.raises(ValueError, match=message):
        PGLSLoss()(**arguments)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("predictions", torch.ones(3, 1, dtype=torch.float16), "predictions.*dtype"),
        ("targets", torch.ones(3, 1, dtype=torch.float32), "targets.*same dtype"),
        ("covariances", [torch.eye(3, dtype=torch.float32)], "covariance.*same dtype"),
        ("batch", torch.zeros(3, dtype=torch.int32), "batch.*int64"),
    ],
)
def test_loss_rejects_unsupported_or_mixed_dtypes(field, value, message):
    predictions, targets, covariances, batch = _single_tree_pgls_inputs()
    arguments = {
        "predictions": predictions,
        "targets": targets,
        "covariances": covariances,
        "batch": batch,
    }
    arguments[field] = value
    with pytest.raises(ValueError, match=message):
        PGLSLoss()(**arguments)


@pytest.mark.parametrize("field", ["predictions", "targets", "covariances"])
@pytest.mark.parametrize("non_finite", [torch.nan, torch.inf])
def test_loss_rejects_non_finite_values(field, non_finite):
    predictions, targets, covariances, batch = _single_tree_pgls_inputs()
    if field == "covariances":
        covariances[0][0, 0] = non_finite
    else:
        value = predictions if field == "predictions" else targets
        value[0, 0] = non_finite
    with pytest.raises(ValueError, match=f"{field}.*finite"):
        PGLSLoss()(predictions, targets, covariances, batch)


@pytest.mark.parametrize(
    ("covariance", "message"),
    [
        (torch.ones(3, 2, dtype=torch.float64), "square"),
        (
            torch.tensor(
                [[1.0, 0.2, 0.0], [0.1, 1.0, 0.0], [0.0, 0.0, 1.0]],
                dtype=torch.float64,
            ),
            "symmetric",
        ),
        (torch.diag(torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64)), "positive definite"),
        (torch.diag(torch.tensor([1.0, 1.0, -0.1], dtype=torch.float64)), "positive definite"),
        (torch.diag(torch.tensor([1.0, 1.0, 0.9e-6], dtype=torch.float64)), "condition ratio"),
    ],
)
def test_loss_rejects_invalid_covariances_before_solve(monkeypatch, covariance, message):
    predictions, targets, _, batch = _single_tree_pgls_inputs()

    def unexpected_solve(*args, **kwargs):
        raise AssertionError("torch.linalg.solve must not run for invalid covariance input")

    monkeypatch.setattr(torch.linalg, "solve", unexpected_solve)
    with pytest.raises(ValueError, match=message):
        PGLSLoss()(predictions, targets, [covariance], batch)


@pytest.mark.parametrize("ratio", [1e-6, 1.1e-6])
def test_loss_accepts_covariance_condition_ratio_at_or_above_boundary(ratio):
    predictions, targets, _, batch = _single_tree_pgls_inputs()
    covariance = torch.diag(torch.tensor([1.0, 0.5, ratio], dtype=torch.float64))
    assert torch.isfinite(PGLSLoss()(predictions, targets, [covariance], batch))


@pytest.mark.parametrize(
    ("covariances", "batch", "message"),
    [
        ([torch.eye(3, dtype=torch.float64)], torch.tensor([-1, -1, -1]), "non-negative"),
        (
            [torch.eye(1, dtype=torch.float64), torch.eye(2, dtype=torch.float64)],
            torch.tensor([0, 2, 2]),
            "contiguous.*0..K-1",
        ),
        (
            [torch.eye(1, dtype=torch.float64), torch.eye(1, dtype=torch.float64)],
            torch.tensor([0, 0, 0]),
            "covariance count",
        ),
        ([torch.eye(2, dtype=torch.float64)], torch.tensor([0, 0, 0]), "leaf count"),
    ],
)
def test_loss_rejects_invalid_batch_covariance_mapping(covariances, batch, message):
    predictions, targets, _, _ = _single_tree_pgls_inputs()
    with pytest.raises(ValueError, match=message):
        PGLSLoss()(predictions, targets, covariances, batch)


@pytest.mark.parametrize("field", ["targets", "batch", "covariances"])
def test_loss_rejects_device_mismatches(field):
    predictions, targets, covariances, batch = _single_tree_pgls_inputs()
    if field == "targets":
        targets = targets.to("meta")
    elif field == "batch":
        batch = batch.to("meta")
    else:
        covariances = [covariances[0].to("meta")]
    with pytest.raises(ValueError, match=f"{field}.*device"):
        PGLSLoss()(predictions, targets, covariances, batch)


def _single_tree_pgls_inputs(
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], torch.Tensor]:
    """Return three leaves from one tree with a known SPD covariance."""
    predictions = torch.tensor([[0.2], [0.7], [1.4]], dtype=dtype)
    targets = torch.tensor([[0.0], [1.0], [1.8]], dtype=dtype)
    covariances = [
        torch.tensor(
            [[1.0, 0.2, 0.1], [0.2, 1.2, 0.3], [0.1, 0.3, 0.9]],
            dtype=dtype,
        )
    ]
    batch = torch.zeros(3, dtype=torch.long)
    return predictions, targets, covariances, batch


def _multi_tree_pgls_inputs(
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], torch.Tensor]:
    """Return five ordered leaves split across two known SPD covariances."""
    predictions = torch.tensor([[0.1], [0.6], [1.0], [1.5], [2.2]], dtype=dtype)
    targets = torch.tensor([[0.0], [0.8], [1.1], [1.7], [2.0]], dtype=dtype)
    covariances = [
        torch.tensor([[1.0, 0.25], [0.25, 0.8]], dtype=dtype),
        torch.tensor(
            [[1.2, 0.2, 0.1], [0.2, 1.0, 0.15], [0.1, 0.15, 0.7]],
            dtype=dtype,
        ),
    ]
    batch = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
    return predictions, targets, covariances, batch


def _single_trait_pgls_inputs(
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], torch.Tensor]:
    """Return one-trait targets in the supported one-dimensional form."""
    predictions, targets, covariances, batch = _single_tree_pgls_inputs(dtype)
    return predictions, targets[:, 0], covariances, batch


def _multi_trait_pgls_inputs(
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], torch.Tensor]:
    """Return two traits for one tree with a known SPD covariance."""
    predictions, first_targets, covariances, batch = _single_tree_pgls_inputs(dtype)
    predictions = torch.cat((predictions, predictions + 0.4), dim=1)
    targets = torch.cat((first_targets, first_targets * 0.5 - 0.2), dim=1)
    return predictions, targets, covariances, batch


def _one_leaf_pgls_inputs(
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], torch.Tensor]:
    """Return the valid one-leaf boundary case."""
    predictions = torch.tensor([[0.4]], dtype=dtype)
    targets = torch.tensor([[1.1]], dtype=dtype)
    covariances = [torch.tensor([[1.5]], dtype=dtype)]
    batch = torch.zeros(1, dtype=torch.long)
    return predictions, targets, covariances, batch
