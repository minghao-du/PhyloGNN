"""Behavior tests for the masked-attention leaf regressor."""

import pytest
import torch


@pytest.mark.parametrize(
    ("input_dim", "hidden_dim"),
    [(True, 4), (0, 4), (2, False), (2, 0), (2.0, 4)],
)
def test_masked_attention_regressor_rejects_invalid_dimensions(input_dim, hidden_dim, torch_module):
    """Constructor dimensions must be positive integers, excluding booleans."""
    from phylognn.models.masked_attention import MaskedAttentionPhyloRegressor

    with pytest.raises(ValueError, match="input_dim|hidden_dim"):
        MaskedAttentionPhyloRegressor(input_dim, hidden_dim, torch_module.eye(2))


@pytest.mark.parametrize(
    ("laplacian", "exception"),
    [
        ("not-a-tensor", TypeError),
        (None, TypeError),
        ([[1.0, 0.0], [0.0, 1.0]], TypeError),
    ],
)
def test_masked_attention_regressor_rejects_non_tensor_laplacian(laplacian, exception):
    """The model contract requires a tensor Laplacian."""
    from phylognn.models.masked_attention import MaskedAttentionPhyloRegressor

    with pytest.raises(exception, match="leaf_laplacian"):
        MaskedAttentionPhyloRegressor(2, 4, laplacian)


@pytest.mark.parametrize(
    "laplacian",
    [
        pytest.param(torch.eye(2, dtype=torch.float64), id="float64"),
        pytest.param(torch.ones((2, 3)), id="non-square"),
        pytest.param(torch.empty((0, 0)), id="empty"),
        pytest.param(
            torch.tensor([[1.0, float("nan")], [0.0, 1.0]]),
            id="non-finite",
        ),
    ],
)
def test_masked_attention_regressor_rejects_invalid_laplacian(laplacian):
    """Laplacian shape, dtype, size, and values are validated at construction."""
    from phylognn.models.masked_attention import MaskedAttentionPhyloRegressor

    with pytest.raises(ValueError, match="leaf_laplacian"):
        MaskedAttentionPhyloRegressor(2, 4, laplacian)


def test_masked_attention_regressor_pools_only_valid_positions(
    association_position_mask, association_representations, torch_module
):
    """Padding should receive exactly zero attention and valid rows sum to one."""
    from phylognn.association import build_leaf_laplacian
    from phylognn.models.masked_attention import MaskedAttentionPhyloRegressor

    tree = pytest.importorskip("ete3").Tree("((A:1,B:1):1,(C:1,D:1):1,(E:1,F:1):1);")
    model = MaskedAttentionPhyloRegressor(
        input_dim=association_representations.size(-1),
        hidden_dim=5,
        leaf_laplacian=build_leaf_laplacian(tree),
    )

    predictions, attention = model(association_representations, association_position_mask)

    assert predictions.shape == (6,)
    assert attention.shape == (6, 4)
    assert torch_module.equal(
        attention[~association_position_mask],
        torch_module.zeros_like(attention[~association_position_mask]),
    )
    assert torch_module.allclose(attention.sum(dim=1), torch_module.ones(6), atol=1e-6, rtol=0)


def test_masked_attention_regressor_registers_bounded_smoothing_laplacian(torch_module):
    """The smoothing parameter starts at 0.1 and the Laplacian is model state."""
    from phylognn.models.masked_attention import MaskedAttentionPhyloRegressor

    laplacian = torch_module.eye(3, dtype=torch_module.float32)
    model = MaskedAttentionPhyloRegressor(2, 4, laplacian)

    assert "leaf_laplacian" in dict(model.named_buffers())
    assert torch_module.allclose(torch_module.sigmoid(model.raw_alpha), torch_module.tensor(0.1))
    assert model.leaf_laplacian.requires_grad is False


def test_masked_attention_regressor_accepts_bool_convertible_mask(torch_module):
    """Integer masks are converted to boolean masks before attention pooling."""
    from phylognn.models.masked_attention import MaskedAttentionPhyloRegressor

    model = MaskedAttentionPhyloRegressor(2, 3, torch_module.eye(2))
    predictions, attention = model(
        torch_module.ones((2, 3, 2)),
        torch_module.tensor([[1, 0, 1], [0, 1, 0]]),
    )

    assert predictions.shape == (2,)
    assert torch_module.equal(attention[0, 1:2], torch_module.zeros(1))
    assert torch_module.equal(attention[1, [0, 2]], torch_module.zeros(2))


@pytest.mark.parametrize(
    ("representations", "position_mask", "message"),
    [
        ("not-a-tensor", torch.ones((2, 3), dtype=torch.bool), "representations"),
        (torch.ones((2, 3)), torch.ones((2, 3), dtype=torch.bool), "shape"),
        (torch.ones((2, 3, 3)), torch.ones((2, 3), dtype=torch.bool), "input_dim"),
        (torch.ones((2, 3, 2), dtype=torch.long), torch.ones((2, 3), dtype=torch.bool), "dtype"),
        (torch.full((2, 3, 2), float("nan")), torch.ones((2, 3), dtype=torch.bool), "finite"),
        (torch.ones((2, 3, 2)), "not-a-tensor", "position_mask"),
        (torch.ones((2, 3, 2)), torch.ones((2, 2), dtype=torch.bool), "position_mask"),
        (
            torch.ones((2, 3, 2)),
            torch.tensor([[True, False, False], [False, False, False]]),
            "position_mask",
        ),
        (torch.ones((3, 3, 2)), torch.ones((3, 3), dtype=torch.bool), "leaf_laplacian"),
    ],
)
def test_masked_attention_regressor_rejects_invalid_forward_inputs(
    representations, position_mask, message, torch_module
):
    """Forward validation rejects malformed tensors before computing attention."""
    from phylognn.models.masked_attention import MaskedAttentionPhyloRegressor

    model = MaskedAttentionPhyloRegressor(2, 3, torch_module.eye(2))

    with pytest.raises((TypeError, ValueError), match=message):
        model(representations, position_mask)
