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


def test_sparse_query_pgls_composition_supports_forward_and_backward():
    """The ordered species embeddings compose with a differentiable PGLS head."""
    from phylognn.models import PGLSRegressionHead
    from phylognn.training import PGLSLoss

    model = _make_model().train()
    head = PGLSRegressionHead(7, 2)
    representations, position_mask = _inputs()
    features = model.forward_leaf_representations(representations, position_mask)
    predictions = head(features)
    loss = PGLSLoss()(
        predictions,
        torch.randn(3, 2),
        [torch.eye(3)],
        torch.zeros(3, dtype=torch.long),
    )
    loss.backward()

    assert features.shape == (3, 7)
    assert predictions.shape == (3, 2)
    assert torch.isfinite(loss)
    assert model.representation_head[0].weight.grad is not None
    assert head.linear.weight.grad is not None


def test_zero_cnn_blocks_and_softmax_support_ablation():
    """The documented identity-CNN and softmax ablations should remain constructible."""
    model = _make_model(num_cnn_blocks=0, attention_normalization="softmax").eval()
    representations, mask = _inputs()

    embeddings, attention = model.encode_sequences(representations, mask)

    assert len(model.cnn_blocks) == 0
    assert embeddings.shape == (3, 7)
    assert torch.all(attention.masked_select(mask.unsqueeze(1)) > 0)


def test_sparse_encoder_requires_float32_representations_before_raw_encoding():
    """The documented raw representation dtype fails before the adapter runs."""
    model = _make_model().eval()
    representations, mask = _inputs(dtype=torch.float64)
    adapter_calls = []
    hook = model.adapter.register_forward_hook(
        lambda _module, _inputs, _output: adapter_calls.append(True)
    )
    try:
        with pytest.raises(ValueError, match="dtype torch.float32"):
            model.encode_sequences(representations, mask)
    finally:
        hook.remove()

    assert adapter_calls == []


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


@pytest.mark.parametrize("chunk_size", [1, 2, 8])
def test_chunk_size_accepts_positive_non_boolean_integers(chunk_size):
    """The raw position encoder chunk bound is an explicit model setting."""
    model = _make_model(chunk_size=chunk_size)

    assert model.chunk_size == chunk_size


def test_chunk_size_none_preserves_full_batch_default():
    """Omitting chunking keeps the source-compatible full-batch default."""
    assert _make_model().chunk_size is None


@pytest.mark.parametrize("chunk_size", [True, False, 0, -1, 1.5, "2"])
def test_chunk_size_rejects_invalid_controls(chunk_size):
    """Invalid controls fail during construction before model execution."""
    with pytest.raises(ValueError, match="chunk_size"):
        _make_model(chunk_size=chunk_size)


@pytest.mark.parametrize(
    ("representations", "position_mask", "message"),
    [
        (torch.ones(3, 6), torch.ones(3, 6, dtype=torch.bool), "shape"),
        (torch.ones(3, 6, 6, dtype=torch.int64), torch.ones(3, 6, dtype=torch.bool), "dtype"),
        (torch.ones(3, 6, 6), torch.full((3, 6), float("nan")), "position_mask"),
        (torch.ones(3, 6, 6), torch.full((3, 6), 0.5), "position_mask"),
    ],
)
def test_chunk_contract_rejects_invalid_shape_dtype_and_masks(
    representations, position_mask, message
):
    """Global input contracts fail before the raw encoder is consumed."""
    with pytest.raises((TypeError, ValueError), match=message):
        _make_model(chunk_size=2).encode_sequences(representations, position_mask)


def test_chunk_contract_accepts_strict_numeric_binary_masks(
    chunked_sequence_representations, chunked_right_padded_mask
):
    """Finite numeric zero/one masks are accepted and normalized to boolean."""
    model = _make_model(
        input_dim=2,
        leaf_laplacian=torch.eye(4),
        chunk_size=2,
        adapter_rank=3,
        token_dim=4,
        num_queries=2,
        slot_dim=3,
        species_dim=5,
        sequence_hidden_dim=3,
        phylogeny_hidden_dim=3,
    ).eval()
    embeddings, attention = model.encode_sequences(
        chunked_sequence_representations, chunked_right_padded_mask.to(torch.float32)
    )

    assert embeddings.shape[0] == attention.shape[0] == 4


@pytest.mark.parametrize("chunk_size", [1, 3, None, 8])
def test_chunked_raw_encoding_preserves_full_batch_outputs_and_order(
    chunk_size, chunked_sequence_representations, chunked_right_padded_mask
):
    """Chunking only the raw encoder preserves one ordered full-batch result."""
    kwargs = {
        "input_dim": 2,
        "leaf_laplacian": torch.eye(4),
        "adapter_rank": 3,
        "token_dim": 4,
        "num_queries": 2,
        "slot_dim": 3,
        "species_dim": 5,
        "sequence_hidden_dim": 3,
        "phylogeny_hidden_dim": 3,
    }
    torch.manual_seed(7)
    full_model = _make_model(**kwargs).eval()
    chunked_model = _make_model(**kwargs, chunk_size=chunk_size).eval()
    chunked_model.load_state_dict(full_model.state_dict())

    expected_embeddings, expected_attention = full_model.encode_sequences(
        chunked_sequence_representations, chunked_right_padded_mask
    )
    actual_embeddings, actual_attention = chunked_model.encode_sequences(
        chunked_sequence_representations, chunked_right_padded_mask
    )
    expected_predictions, _ = full_model(
        chunked_sequence_representations, chunked_right_padded_mask
    )
    actual_predictions, _ = chunked_model(
        chunked_sequence_representations, chunked_right_padded_mask
    )

    assert torch.allclose(actual_embeddings, expected_embeddings, atol=1e-6, rtol=0)
    assert torch.allclose(actual_attention, expected_attention, atol=1e-6, rtol=0)
    assert torch.allclose(actual_predictions, expected_predictions, atol=1e-6, rtol=0)
    assert torch.equal(
        actual_attention.masked_select(~chunked_right_padded_mask.unsqueeze(1)),
        torch.zeros_like(actual_attention.masked_select(~chunked_right_padded_mask.unsqueeze(1))),
    )


def test_chunked_sparse_encoder_bounds_raw_work_and_runs_downstream_once(
    chunked_sequence_representations, chunked_right_padded_mask
):
    """Chunking retains one complete-batch attention, pooling, and prediction pass."""
    model = _make_model(
        input_dim=2,
        leaf_laplacian=torch.eye(4),
        adapter_rank=3,
        token_dim=4,
        num_queries=2,
        slot_dim=3,
        species_dim=5,
        sequence_hidden_dim=3,
        phylogeny_hidden_dim=3,
        chunk_size=3,
    ).eval()
    adapter_batch_sizes = []
    cnn_batch_sizes = []
    attention_batch_sizes = []
    pooling_batch_sizes = []
    prediction_batch_sizes = []
    phylogeny_batch_sizes = []
    hooks = [
        model.adapter.register_forward_hook(
            lambda _module, inputs, _output: adapter_batch_sizes.append(inputs[0].size(0))
        ),
        model.cnn_blocks[0].register_forward_hook(
            lambda _module, inputs, _output: cnn_batch_sizes.append(inputs[0].size(0))
        ),
        *[
            projection.register_forward_hook(
                lambda _module, inputs, _output: attention_batch_sizes.append(inputs[0].size(0))
            )
            for projection in model.key_projections
        ],
        model.representation_head.register_forward_hook(
            lambda _module, inputs, _output: pooling_batch_sizes.append(inputs[0].size(0))
        ),
        model.sequence_head.register_forward_hook(
            lambda _module, inputs, _output: prediction_batch_sizes.append(inputs[0].size(0))
        ),
        model.phylogeny_output.register_forward_hook(
            lambda _module, inputs, _output: phylogeny_batch_sizes.append(inputs[0].size(0))
        ),
    ]
    try:
        model(chunked_sequence_representations, chunked_right_padded_mask)
    finally:
        for hook in hooks:
            hook.remove()

    assert adapter_batch_sizes == [3, 1]
    assert cnn_batch_sizes == [3, 1]
    assert attention_batch_sizes == [4, 4]
    assert pooling_batch_sizes == [4]
    assert prediction_batch_sizes == [4]
    assert phylogeny_batch_sizes == [4]


def test_chunked_sparse_encoder_checks_finiteness_and_raw_work_per_chunk(
    chunked_sequence_representations, chunked_right_padded_mask, monkeypatch
):
    """Finiteness, adapter, and CNN work never consume more than one raw chunk."""
    model = _make_model(
        input_dim=2,
        leaf_laplacian=torch.eye(4),
        adapter_rank=3,
        token_dim=4,
        num_queries=2,
        slot_dim=3,
        species_dim=5,
        sequence_hidden_dim=3,
        phylogeny_hidden_dim=3,
        chunk_size=3,
    ).eval()
    finite_check_batch_sizes = []
    adapter_batch_sizes = []
    cnn_batch_sizes = []
    original_isfinite = torch.isfinite

    def record_finiteness(tensor):
        if tensor.ndim == 3:
            finite_check_batch_sizes.append(tensor.size(0))
        return original_isfinite(tensor)

    monkeypatch.setattr(torch, "isfinite", record_finiteness)
    hooks = [
        model.adapter.register_forward_hook(
            lambda _module, inputs, _output: adapter_batch_sizes.append(inputs[0].size(0))
        ),
        model.cnn_blocks[0].register_forward_hook(
            lambda _module, inputs, _output: cnn_batch_sizes.append(inputs[0].size(0))
        ),
    ]
    try:
        model.encode_sequences(chunked_sequence_representations, chunked_right_padded_mask)
    finally:
        for hook in hooks:
            hook.remove()

    assert finite_check_batch_sizes == [3, 1]
    assert adapter_batch_sizes == [3, 1]
    assert cnn_batch_sizes == [3, 1]


def test_sparse_encoder_rejects_nonfinite_values_when_their_chunk_is_consumed(
    chunked_sequence_representations, chunked_right_padded_mask, monkeypatch
):
    """A late invalid chunk fails before that chunk reaches the raw encoder."""
    model = _make_model(
        input_dim=2,
        leaf_laplacian=torch.eye(4),
        adapter_rank=3,
        token_dim=4,
        num_queries=2,
        slot_dim=3,
        species_dim=5,
        sequence_hidden_dim=3,
        phylogeny_hidden_dim=3,
        chunk_size=3,
    ).eval()
    representations = chunked_sequence_representations.clone()
    representations[-1, 0, 0] = float("nan")
    finite_check_batch_sizes = []
    adapter_batch_sizes = []
    original_isfinite = torch.isfinite

    def record_finiteness(tensor):
        if tensor.ndim == 3:
            finite_check_batch_sizes.append(tensor.size(0))
        return original_isfinite(tensor)

    monkeypatch.setattr(torch, "isfinite", record_finiteness)
    hook = model.adapter.register_forward_hook(
        lambda _module, inputs, _output: adapter_batch_sizes.append(inputs[0].size(0))
    )
    try:
        with pytest.raises(ValueError, match="representations.*finite"):
            model.encode_sequences(representations, chunked_right_padded_mask)
    finally:
        hook.remove()

    assert finite_check_batch_sizes == [3, 1]
    assert adapter_batch_sizes == [3]


@pytest.mark.parametrize(
    ("position_mask", "message"),
    [
        (torch.ones((4, 3), dtype=torch.bool), "representations"),
        (
            torch.tensor(
                [
                    [True, True, True],
                    [True, True, False],
                    [True, False, False],
                    [False, False, False],
                ]
            ),
            "position_mask",
        ),
        (
            torch.tensor(
                [[True, True, True], [True, False, True], [True, False, False], [True, True, True]]
            ),
            "position_mask",
        ),
        (torch.full((4, 3), float("inf")), "position_mask"),
        (torch.full((4, 3), 0.5), "position_mask"),
    ],
)
def test_sparse_chunk_contract_rejects_empty_and_invalid_mask_rows(
    chunked_sequence_representations, position_mask, message
):
    """Global representation and mask contracts fail before raw encoding starts."""
    model = _make_model(
        input_dim=2,
        leaf_laplacian=torch.eye(4),
        adapter_rank=3,
        token_dim=4,
        num_queries=2,
        slot_dim=3,
        species_dim=5,
        sequence_hidden_dim=3,
        phylogeny_hidden_dim=3,
        chunk_size=2,
    ).eval()
    representations = chunked_sequence_representations
    if message == "representations":
        representations = representations[:0]
        position_mask = position_mask[:0]

    with pytest.raises((TypeError, ValueError), match=message):
        model.encode_sequences(representations, position_mask)


@pytest.mark.parametrize(
    ("representations", "position_mask", "message"),
    [
        (torch.ones((3, 3, 2)), torch.ones((3, 3), dtype=torch.bool), "leaf count"),
        (torch.ones((4, 3, 3)), torch.ones((4, 3), dtype=torch.bool), "input_dim"),
    ],
)
def test_sparse_encoder_rejects_global_contracts_before_adapter(
    representations, position_mask, message
):
    """Leaf count and shape validation must precede every raw encoder call."""
    model = _make_model(
        input_dim=2,
        leaf_laplacian=torch.eye(4),
        adapter_rank=3,
        token_dim=4,
        num_queries=2,
        slot_dim=3,
        species_dim=5,
        sequence_hidden_dim=3,
        phylogeny_hidden_dim=3,
        chunk_size=2,
    ).eval()
    adapter_calls = []
    hook = model.adapter.register_forward_hook(
        lambda _module, _inputs, _output: adapter_calls.append(True)
    )
    try:
        with pytest.raises(ValueError, match=message):
            model.encode_sequences(representations, position_mask)
    finally:
        hook.remove()

    assert adapter_calls == []


def test_sparse_encoder_rejects_laplacian_device_mismatch_before_adapter():
    """The graph operator must share the raw representation device before encoding."""
    model = _make_model(
        input_dim=2,
        leaf_laplacian=torch.eye(4),
        adapter_rank=3,
        token_dim=4,
        num_queries=2,
        slot_dim=3,
        species_dim=5,
        sequence_hidden_dim=3,
        phylogeny_hidden_dim=3,
        chunk_size=2,
    ).eval()
    model._buffers["leaf_laplacian"] = torch.eye(4, device="meta")
    representations = torch.ones((4, 3, 2))
    position_mask = torch.ones((4, 3), dtype=torch.bool)
    adapter_calls = []
    hook = model.adapter.register_forward_hook(
        lambda _module, _inputs, _output: adapter_calls.append(True)
    )
    try:
        with pytest.raises(ValueError, match="same device.*leaf_laplacian"):
            model.encode_sequences(representations, position_mask)
    finally:
        hook.remove()

    assert adapter_calls == []


def test_sparse_encoder_rejects_laplacian_dtype_mismatch_before_adapter():
    """The graph operator must share the raw representation dtype before encoding."""
    model = _make_model(
        input_dim=2,
        leaf_laplacian=torch.eye(4),
        adapter_rank=3,
        token_dim=4,
        num_queries=2,
        slot_dim=3,
        species_dim=5,
        sequence_hidden_dim=3,
        phylogeny_hidden_dim=3,
        chunk_size=2,
    ).eval()
    model._buffers["leaf_laplacian"] = torch.eye(4, dtype=torch.float64)
    representations = torch.ones((4, 3, 2))
    position_mask = torch.ones((4, 3), dtype=torch.bool)
    adapter_calls = []
    hook = model.adapter.register_forward_hook(
        lambda _module, _inputs, _output: adapter_calls.append(True)
    )
    try:
        with pytest.raises(ValueError, match="same dtype.*leaf_laplacian"):
            model.encode_sequences(representations, position_mask)
    finally:
        hook.remove()

    assert adapter_calls == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA for device mismatch")
def test_chunk_contract_rejects_mask_on_different_device():
    """Representations and masks must share a device before chunk execution."""
    model = _make_model().cuda()
    representations, mask = _inputs()

    with pytest.raises(ValueError, match="same device|device"):
        model.encode_sequences(representations.cuda(), mask)


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
