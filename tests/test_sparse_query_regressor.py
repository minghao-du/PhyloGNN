"""Behavior tests for the sparse-query phylogenetic regressor."""

import math

import pytest
import torch


def _make_model(**overrides):
    """Build a compact deterministic model for contract tests."""
    from phylognn.models.sparse_query import SparseQueryPhyloRegressor

    kwargs = {
        "input_dim": 6,
        "leaf_laplacian": torch.eye(3),
        "adapter_rank": 4,
        "token_dim": 8,
        "num_cnn_blocks": 1,
        "cnn_kernel_sizes": (3, 5),
        "num_queries": 3,
        "slot_dim": 4,
        "species_dim": 7,
        "sequence_hidden_dim": 5,
        "phylogeny_hidden_dim": 6,
        "adapter_dropout_prob": 0.0,
        "cnn_dropout_prob": 0.0,
        "representation_dropout_prob": 0.0,
        "sequence_dropout_prob": 0.0,
        "phylogeny_dropout_prob": 0.0,
    }
    kwargs.update(overrides)
    return SparseQueryPhyloRegressor(**kwargs)


def _inputs(dtype=torch.float32):
    representations = torch.randn(3, 6, 6, dtype=dtype)
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0, 0],
        ],
        dtype=torch.bool,
    )
    return representations, mask


def test_encode_sequences_returns_normalized_query_attention():
    """Every query should pool valid positions into one species embedding."""
    model = _make_model().eval()
    representations, mask = _inputs()

    embeddings, attention = model.encode_sequences(representations, mask)

    assert embeddings.shape == (3, 7)
    assert attention.shape == (3, 3, 6)
    assert torch.equal(
        attention.masked_select(~mask.unsqueeze(1)),
        torch.zeros_like(attention.masked_select(~mask.unsqueeze(1))),
    )
    assert torch.allclose(attention.sum(dim=-1), torch.ones(3, 3), atol=1e-6, rtol=0)


def test_padding_values_do_not_change_embeddings_or_attention():
    """Masked padding must not leak through the adapter or local convolutions."""
    model = _make_model().eval()
    representations, mask = _inputs()
    changed_padding = representations.clone()
    changed_padding[~mask] = 1_000_000.0

    expected = model.encode_sequences(representations, mask)
    actual = model.encode_sequences(changed_padding, mask)

    assert torch.allclose(actual[0], expected[0], atol=1e-6, rtol=0)
    assert torch.allclose(actual[1], expected[1], atol=1e-6, rtol=0)


def test_queries_have_independent_key_value_and_query_parameters():
    """No query should share its relevance or effect projection with another."""
    model = _make_model(num_queries=4)

    assert len(model.key_projections) == 4
    assert len(model.value_projections) == 4
    assert len(model.query_vectors) == 4
    assert len({id(module) for module in model.key_projections}) == 4
    assert len({id(module) for module in model.value_projections}) == 4
    assert len({parameter.data_ptr() for parameter in model.query_vectors}) == 4


def test_predict_from_embeddings_returns_gated_components():
    """The final prediction must be the gated sum of inspectable branches."""
    model = _make_model(phylogeny_gate_init=0.05).eval()
    embeddings = torch.randn(3, 7)

    prediction, sequence_prediction, phylogeny_prediction = model.predict_from_embeddings(
        embeddings
    )

    assert prediction.shape == sequence_prediction.shape == phylogeny_prediction.shape == (3,)
    expected = sequence_prediction + torch.sigmoid(model.raw_beta) * phylogeny_prediction
    assert torch.allclose(prediction, expected, atol=1e-7, rtol=0)
    assert torch.allclose(torch.sigmoid(model.raw_beta), torch.tensor(0.05), atol=1e-7, rtol=0)


def test_forward_matches_two_stage_api():
    """The convenience forward path should preserve the composable API exactly."""
    model = _make_model().eval()
    representations, mask = _inputs()

    prediction, attention = model(representations, mask)
    embeddings, expected_attention = model.encode_sequences(representations, mask)
    expected_prediction = model.predict_from_embeddings(embeddings)[0]

    assert torch.equal(prediction, expected_prediction)
    assert torch.equal(attention, expected_attention)


def test_zero_cnn_blocks_and_softmax_support_ablation():
    """The documented identity-CNN and softmax ablations should remain constructible."""
    model = _make_model(num_cnn_blocks=0, attention_normalization="softmax").eval()
    representations, mask = _inputs()

    embeddings, attention = model.encode_sequences(representations, mask)

    assert len(model.cnn_blocks) == 0
    assert embeddings.shape == (3, 7)
    assert torch.all(attention.masked_select(mask.unsqueeze(1)) > 0)


def test_floating_dtypes_are_supported_without_silent_conversion():
    """Moving the model to float64 should preserve float64 through both stages."""
    model = _make_model(leaf_laplacian=torch.eye(3, dtype=torch.float64)).double().eval()
    representations, mask = _inputs(dtype=torch.float64)

    embeddings, attention = model.encode_sequences(representations, mask)
    prediction = model.predict_from_embeddings(embeddings)[0]

    assert embeddings.dtype == torch.float64
    assert attention.dtype == torch.float64
    assert prediction.dtype == torch.float64


@pytest.mark.parametrize(
    "mask",
    [
        pytest.param(torch.tensor([[1, 2], [1, 0], [1, 0]]), id="non-binary"),
        pytest.param(torch.tensor([[1, 0], [0, 0], [1, 0]]), id="empty-row"),
        pytest.param(torch.tensor([[1, 0, 1], [1, 1, 0], [1, 0, 0]]), id="internal-gap"),
    ],
)
def test_position_mask_must_be_binary_nonempty_right_padding(mask):
    """Malformed masks should fail before sequence operations run."""
    model = _make_model()
    representations = torch.randn(3, mask.size(1), 6)

    with pytest.raises(ValueError, match="position_mask"):
        model.encode_sequences(representations, mask)


def test_binary_numeric_position_mask_is_accepted():
    """Numeric masks remain convenient when they strictly contain zero and one."""
    model = _make_model().eval()
    representations, mask = _inputs()

    embeddings, attention = model.encode_sequences(representations, mask.to(torch.float32))

    assert embeddings.shape == (3, 7)
    assert torch.isfinite(attention).all()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"input_dim": 0}, "input_dim"),
        ({"num_queries": True}, "num_queries"),
        ({"num_cnn_blocks": -1}, "num_cnn_blocks"),
        ({"cnn_kernel_sizes": (3, 4)}, "cnn_kernel_sizes"),
        ({"adapter_dropout_prob": 1.0}, "adapter_dropout_prob"),
        ({"attention_normalization": "sparsemax"}, "attention_normalization"),
        ({"phylogeny_gate_init": 0.0}, "phylogeny_gate_init"),
    ],
)
def test_constructor_rejects_invalid_configuration(overrides, message):
    """Every configurable architecture value should have an explicit contract."""
    with pytest.raises((TypeError, ValueError), match=message):
        _make_model(**overrides)


@pytest.mark.parametrize(
    "laplacian",
    [
        pytest.param([[1.0]], id="not-tensor"),
        pytest.param(torch.ones(2, 3), id="not-square"),
        pytest.param(torch.ones(2, 2, dtype=torch.long), id="not-floating"),
        pytest.param(torch.tensor([[1.0, math.nan], [0.0, 1.0]]), id="non-finite"),
    ],
)
def test_constructor_rejects_invalid_laplacian(laplacian):
    """The fixed graph operator must be a finite floating square tensor."""
    with pytest.raises((TypeError, ValueError), match="leaf_laplacian"):
        _make_model(leaf_laplacian=laplacian)


def test_predict_from_embeddings_requires_all_laplacian_leaves():
    """Graph prediction must receive one embedding per fixed Laplacian row."""
    model = _make_model()

    with pytest.raises(ValueError, match="leaf_laplacian"):
        model.predict_from_embeddings(torch.randn(2, 7))
