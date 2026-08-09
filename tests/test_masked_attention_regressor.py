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
    leaf_regression_position_mask,
    leaf_regression_representations,
    leaf_regression_targets,
    leaf_regression_tree,
    torch_module,
):
    """Padding should receive exactly zero attention and valid rows sum to one."""
    from phylognn.leaf_regression import prepare_leaf_regression
    from phylognn.models.masked_attention import MaskedAttentionPhyloRegressor

    data = prepare_leaf_regression(
        leaf_regression_tree,
        leaf_regression_representations,
        leaf_regression_position_mask,
        leaf_regression_targets,
    )
    model = MaskedAttentionPhyloRegressor(
        input_dim=data.representations.size(-1),
        hidden_dim=5,
        leaf_laplacian=data.leaf_laplacian,
    )

    predictions, attention = model(data.representations, data.position_mask)

    assert predictions.shape == (6,)
    assert attention.shape == (6, 4)
    assert torch_module.equal(
        attention[~data.position_mask],
        torch_module.zeros_like(attention[~data.position_mask]),
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


@pytest.mark.parametrize("dropout_prob", [-0.1, 1.0, float("nan")])
def test_masked_attention_regressor_rejects_invalid_dropout(dropout_prob, torch_module):
    """Dropout probability follows PyTorch's [0, 1) contract."""
    from phylognn.models.masked_attention import MaskedAttentionPhyloRegressor

    with pytest.raises(ValueError, match="dropout_prob"):
        MaskedAttentionPhyloRegressor(2, 4, torch_module.eye(2), dropout_prob=dropout_prob)


def test_masked_attention_regressor_applies_dropout_only_in_training(torch_module):
    """Projected representations are stochastic in training and deterministic in evaluation."""
    from phylognn.models.masked_attention import MaskedAttentionPhyloRegressor

    model = MaskedAttentionPhyloRegressor(2, 4, torch_module.eye(2), dropout_prob=0.5)
    representations = torch_module.ones((2, 3, 2))
    position_mask = torch_module.ones((2, 3), dtype=torch_module.bool)

    model.eval()
    eval_output = model(representations, position_mask)[0]
    assert torch_module.equal(eval_output, model(representations, position_mask)[0])

    model.train()
    train_outputs = [model(representations, position_mask)[0] for _ in range(4)]
    assert any(not torch_module.equal(train_outputs[0], output) for output in train_outputs[1:])


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


@pytest.mark.parametrize(
    ("attention_normalization", "exception_type"),
    [
        ("sparsemax", ValueError),
        ("Softmax", ValueError),
        ("ENTMAX15", ValueError),
        ("", ValueError),
        (None, TypeError),
        (1, TypeError),
        (True, TypeError),
        (["softmax"], TypeError),
    ],
)
def test_masked_attention_regressor_rejects_invalid_attention_normalization(
    attention_normalization, exception_type, torch_module
):
    """Unsupported normalization selections fail with an actionable contract."""
    from phylognn.models.masked_attention import MaskedAttentionPhyloRegressor

    with pytest.raises(exception_type) as exc_info:
        MaskedAttentionPhyloRegressor(
            2,
            4,
            torch_module.eye(2),
            attention_normalization=attention_normalization,
        )

    message = str(exc_info.value)
    assert "attention_normalization" in message
    assert "softmax" in message
    assert "entmax15" in message


# ---------------------------------------------------------------------------
# T002: Deterministic model tests parameterized across softmax and entmax15
# ---------------------------------------------------------------------------


def _build_deterministic_model(
    n_leaves: int,
    hidden_dim: int,
    attention_normalization: str,
    *,
    scorer_weight: float,
    scorer_bias_values: list[float] | None = None,
) -> torch.nn.Module:
    """Return a no-dropout evaluation-mode model with deterministic scorer."""
    from phylognn.models.masked_attention import MaskedAttentionPhyloRegressor

    laplacian = torch.eye(n_leaves, dtype=torch.float32)
    model = MaskedAttentionPhyloRegressor(
        input_dim=2,
        hidden_dim=hidden_dim,
        leaf_laplacian=laplacian,
        dropout_prob=0.0,
        attention_normalization=attention_normalization,
    )
    model.eval()
    # Set scorer to deterministic values so attention is predictable.
    with torch.no_grad():
        model.attention_scorer.weight.fill_(scorer_weight)
        if scorer_bias_values is not None:
            model.attention_scorer.bias.copy_(torch.tensor(scorer_bias_values))
        else:
            model.attention_scorer.bias.fill_(0.0)
    return model


@pytest.mark.parametrize("mode", ["softmax", "entmax15"])
class TestMaskedAttentionNormalizationModes:
    """Deterministic tests shared across both attention normalization modes."""

    def test_inspectable_mode_selection(self, mode, torch_module):
        """The model exposes the selected normalization mode string."""
        from phylognn.models.masked_attention import MaskedAttentionPhyloRegressor

        model = MaskedAttentionPhyloRegressor(
            2, 4, torch_module.eye(2), attention_normalization=mode
        )
        assert model.attention_normalization == mode

    def test_attention_nonnegative(self, mode, torch_module):
        """Attention weights must be non-negative for both modes."""
        model = _build_deterministic_model(2, 4, mode, scorer_weight=1.0)
        representations = torch.randn(2, 3, 2)
        mask = torch.ones(2, 3, dtype=torch.bool)
        _, attention = model(representations, mask)
        assert (attention >= 0).all()

    def test_exact_padding_zeros(self, mode, torch_module):
        """Padding positions must have exactly zero attention."""
        model = _build_deterministic_model(2, 4, mode, scorer_weight=1.0)
        representations = torch.randn(2, 3, 2)
        mask = torch.tensor([[True, True, False], [True, False, False]])
        _, attention = model(representations, mask)
        assert torch.equal(attention[0, 2:], torch.zeros(1))
        assert torch.equal(attention[1, 1:], torch.zeros(2))

    def test_row_sums_within_tolerance(self, mode, torch_module):
        """Each row must sum to 1 within 1e-6."""
        model = _build_deterministic_model(3, 4, mode, scorer_weight=1.0)
        representations = torch.randn(3, 4, 2)
        mask = torch.ones(3, 4, dtype=torch.bool)
        _, attention = model(representations, mask)
        assert torch.allclose(attention.sum(dim=1), torch.ones(3), atol=1e-6, rtol=0)

    def test_finite_outputs(self, mode, torch_module):
        """Both predictions and attention must be finite."""
        model = _build_deterministic_model(2, 4, mode, scorer_weight=1.0)
        representations = torch.randn(2, 3, 2)
        mask = torch.ones(2, 3, dtype=torch.bool)
        predictions, attention = model(representations, mask)
        assert torch.isfinite(predictions).all()
        assert torch.isfinite(attention).all()

    def test_single_valid_position_gets_full_weight(self, mode, torch_module):
        """A row with exactly one valid position assigns it weight 1."""
        model = _build_deterministic_model(2, 4, mode, scorer_weight=1.0)
        representations = torch.randn(2, 3, 2)
        mask = torch.tensor([[True, False, False], [False, True, False]])
        _, attention = model(representations, mask)
        assert torch.allclose(attention[0, 0], torch.tensor(1.0), atol=1e-6)
        assert torch.allclose(attention[1, 1], torch.tensor(1.0), atol=1e-6)

    def test_tied_scores(self, mode, torch_module):
        """When scores are tied, attention should be uniform over valid positions."""
        model = _build_deterministic_model(2, 4, mode, scorer_weight=0.0)
        # All positions will have same bias -> tied scores
        representations = torch.zeros(2, 3, 2)
        mask = torch.tensor([[True, True, True], [True, True, False]])
        _, attention = model(representations, mask)
        # Row 0 has 3 valid -> each ~1/3; row 1 has 2 valid -> each ~1/2
        assert torch.allclose(attention[0, :3], torch.tensor([1 / 3, 1 / 3, 1 / 3]), atol=1e-5)
        assert torch.allclose(attention[1, :2], torch.tensor([1 / 2, 1 / 2]), atol=1e-5)

    @pytest.mark.parametrize("score_value", [-1e4, 0.0, 1e4])
    def test_representative_finite_attention_scores(self, mode, score_value, torch_module):
        """Finite scores at -1e4, 0, and 1e4 produce finite attention."""
        model = _build_deterministic_model(
            2, 4, mode, scorer_weight=0.0, scorer_bias_values=[score_value]
        )
        representations = torch.zeros(2, 3, 2)
        mask = torch.ones(2, 3, dtype=torch.bool)
        predictions, attention = model(representations, mask)
        assert torch.isfinite(attention).all()
        assert torch.isfinite(predictions).all()

    def test_boolean_convertible_masks(self, mode, torch_module):
        """Integer masks are handled correctly as in the base tests."""
        model = _build_deterministic_model(2, 4, mode, scorer_weight=1.0)
        representations = torch.randn(2, 3, 2)
        int_mask = torch.tensor([[1, 0, 1], [0, 1, 0]])
        predictions, attention = model(representations, int_mask)
        assert predictions.shape == (2,)
        assert torch.equal(attention[0, 1:2], torch.zeros(1))
        assert torch.equal(attention[1, [0, 2]], torch.zeros(2))

    def test_independently_recomputed_downstream(self, mode, torch_module):
        """Independently recompute pooling, Laplacian smoothing, and regression head."""
        n_leaves = 3
        hidden_dim = 4
        model = _build_deterministic_model(n_leaves, hidden_dim, mode, scorer_weight=1.0)
        representations = torch.randn(n_leaves, 3, 2)
        mask = torch.ones(n_leaves, 3, dtype=torch.bool)

        # Run model
        predictions, attention = model(representations, mask)

        # Independently recompute from attention
        with torch.no_grad():
            projected = torch.tanh(model.position_projection(representations))
            pooled = (attention.unsqueeze(-1) * projected).sum(dim=1)
            alpha = torch.sigmoid(model.raw_alpha)
            smoothed = pooled - alpha * (model.leaf_laplacian @ pooled)
            expected_predictions = model.regression_head(smoothed).squeeze(-1)

        assert torch.allclose(predictions, expected_predictions, atol=1e-6)


def test_entmax15_sparse_valid_support(torch_module):
    """entmax15 produces at least one exact zero at a valid position with spread scores."""
    model = _build_deterministic_model(2, 4, "entmax15", scorer_weight=1.0)
    # Use inputs designed to produce spread scores -> some valid zeros
    # Set different projection weights so position scores diverge
    with torch.no_grad():
        model.position_projection.weight.copy_(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
        )
        model.position_projection.bias.fill_(0.0)
        model.attention_scorer.weight.copy_(torch.tensor([[2.0, -2.0, 1.0, -1.0]]))
        model.attention_scorer.bias.fill_(0.0)

    # Representations designed so positions get very different scores
    representations = torch.tensor(
        [
            [[3.0, 0.0], [-3.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [[0.0, 3.0], [0.0, -3.0], [0.0, 0.0], [0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    mask = torch.ones(2, 4, dtype=torch.bool)
    _, attention = model(representations, mask)

    # With entmax15 and spread scores, at least one valid position should get exact 0
    has_zero_at_valid = (attention == 0.0).any(dim=1)
    assert has_zero_at_valid.any(), "entmax15 should produce sparse attention with spread scores"
    # Confirm all rows still sum to 1
    assert torch.allclose(attention.sum(dim=1), torch.ones(2), atol=1e-6)


# ---------------------------------------------------------------------------
# T007: Softmax compatibility regression tests (User Story 2)
# ---------------------------------------------------------------------------


def _compute_prefeature_reference(
    model: torch.nn.Module,
    representations: torch.Tensor,
    position_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independently compute the pre-feature masked-softmax pipeline.

    This replicates the original mathematical reference *without* calling
    ``model.forward``, using only the model's stored parameters and buffers
    to verify softmax-path correctness.

    Steps:
        1. Project: ``tanh(X @ W_proj.T + b_proj)``
        2. Score:   ``H_proj @ W_score.T + b_score``
        3. Mask:    replace invalid positions with ``-inf``
        4. Softmax: row-wise ``exp / sum(exp)``
        5. Post-mask: multiply by boolean mask
        6. Pool:    ``sum(attention * projected, dim=1)``
        7. Smooth:  ``pooled - sigmoid(raw_alpha) * (L @ pooled)``
        8. Head:    ``smoothed @ W_head.T + b_head``
    """
    with torch.no_grad():
        mask = position_mask.to(dtype=torch.bool)

        # 1. Projection (no dropout in eval mode)
        projected = torch.tanh(model.position_projection(representations))

        # 2. Scoring
        scores = model.attention_scorer(projected).squeeze(-1)

        # 3. Pre-mask
        masked_scores = scores.masked_fill(~mask, float("-inf"))

        # 4. Row-wise softmax
        ref_attention = torch.softmax(masked_scores, dim=1)

        # 5. Post-mask (exact zeros at invalid positions)
        ref_attention = ref_attention * mask.to(dtype=ref_attention.dtype)

        # 6. Attention pooling
        pooled = (ref_attention.unsqueeze(-1) * projected).sum(dim=1)

        # 7. Laplacian smoothing
        alpha = torch.sigmoid(model.raw_alpha)
        smoothed = pooled - alpha * (model.leaf_laplacian @ pooled)

        # 8. Regression head
        ref_predictions = model.regression_head(smoothed).squeeze(-1)

    return ref_predictions, ref_attention


class TestSoftmaxCompatibilityRegression:
    """T007: Verify omitted and explicit softmax produce identical outputs
    matching the independent pre-feature mathematical reference."""

    N_LEAVES = 4
    HIDDEN_DIM = 5
    INPUT_DIM = 3
    N_POSITIONS = 6

    @pytest.fixture()
    def shared_state(self, torch_module):
        """Build a deterministic model and return its state dict and inputs."""
        from phylognn.models.masked_attention import MaskedAttentionPhyloRegressor

        laplacian = torch_module.eye(self.N_LEAVES, dtype=torch_module.float32)
        # Build with explicit softmax to capture a reference state dict.
        model = MaskedAttentionPhyloRegressor(
            input_dim=self.INPUT_DIM,
            hidden_dim=self.HIDDEN_DIM,
            leaf_laplacian=laplacian,
            dropout_prob=0.0,
            attention_normalization="softmax",
        )
        model.eval()
        # Deterministic weights for reproducibility.
        torch_module.manual_seed(42)
        for p in model.parameters():
            p.data.uniform_(-0.5, 0.5)
        state_dict = model.state_dict()

        # Deterministic inputs with a non-trivial mask.
        torch_module.manual_seed(7)
        representations = torch_module.randn(self.N_LEAVES, self.N_POSITIONS, self.INPUT_DIM)
        position_mask = torch_module.tensor(
            [
                [True, True, True, False, False, False],
                [True, True, True, True, False, False],
                [True, False, False, False, False, False],
                [True, True, True, True, True, True],
            ]
        )
        return state_dict, laplacian, representations, position_mask

    def _build_model(self, torch_module, laplacian, state_dict, **kwargs):
        """Construct a model, load the shared state, and set to eval."""
        from phylognn.models.masked_attention import MaskedAttentionPhyloRegressor

        model = MaskedAttentionPhyloRegressor(
            input_dim=self.INPUT_DIM,
            hidden_dim=self.HIDDEN_DIM,
            leaf_laplacian=laplacian,
            dropout_prob=0.0,
            **kwargs,
        )
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def test_omitted_mode_exposes_softmax(self, shared_state, torch_module):
        """Model with omitted attention_normalization reports 'softmax'."""
        state_dict, laplacian, _, _ = shared_state
        model = self._build_model(torch_module, laplacian, state_dict)
        assert model.attention_normalization == "softmax"

    def test_explicit_mode_exposes_softmax(self, shared_state, torch_module):
        """Model with explicit 'softmax' reports 'softmax'."""
        state_dict, laplacian, _, _ = shared_state
        model = self._build_model(
            torch_module,
            laplacian,
            state_dict,
            attention_normalization="softmax",
        )
        assert model.attention_normalization == "softmax"

    def test_omitted_matches_prefeature_reference_attention(
        self,
        shared_state,
        torch_module,
    ):
        """Omitted-mode attention matches the independent softmax reference."""
        state_dict, laplacian, representations, position_mask = shared_state
        model = self._build_model(torch_module, laplacian, state_dict)
        _, attention = model(representations, position_mask)
        _, ref_attention = _compute_prefeature_reference(
            model,
            representations,
            position_mask,
        )
        assert torch_module.allclose(attention, ref_attention, atol=1e-6, rtol=0)

    def test_explicit_matches_prefeature_reference_attention(
        self,
        shared_state,
        torch_module,
    ):
        """Explicit-softmax attention matches the independent softmax reference."""
        state_dict, laplacian, representations, position_mask = shared_state
        model = self._build_model(
            torch_module,
            laplacian,
            state_dict,
            attention_normalization="softmax",
        )
        _, attention = model(representations, position_mask)
        _, ref_attention = _compute_prefeature_reference(
            model,
            representations,
            position_mask,
        )
        assert torch_module.allclose(attention, ref_attention, atol=1e-6, rtol=0)

    def test_omitted_matches_prefeature_reference_predictions(
        self,
        shared_state,
        torch_module,
    ):
        """Omitted-mode predictions match the independent softmax reference."""
        state_dict, laplacian, representations, position_mask = shared_state
        model = self._build_model(torch_module, laplacian, state_dict)
        predictions, _ = model(representations, position_mask)
        ref_predictions, _ = _compute_prefeature_reference(
            model,
            representations,
            position_mask,
        )
        assert torch_module.allclose(predictions, ref_predictions, atol=1e-6, rtol=0)

    def test_explicit_matches_prefeature_reference_predictions(
        self,
        shared_state,
        torch_module,
    ):
        """Explicit-softmax predictions match the independent softmax reference."""
        state_dict, laplacian, representations, position_mask = shared_state
        model = self._build_model(
            torch_module,
            laplacian,
            state_dict,
            attention_normalization="softmax",
        )
        predictions, _ = model(representations, position_mask)
        ref_predictions, _ = _compute_prefeature_reference(
            model,
            representations,
            position_mask,
        )
        assert torch_module.allclose(predictions, ref_predictions, atol=1e-6, rtol=0)

    def test_omitted_and_explicit_produce_identical_attention(
        self,
        shared_state,
        torch_module,
    ):
        """Default and explicit softmax models produce identical attention."""
        state_dict, laplacian, representations, position_mask = shared_state
        model_default = self._build_model(torch_module, laplacian, state_dict)
        model_explicit = self._build_model(
            torch_module,
            laplacian,
            state_dict,
            attention_normalization="softmax",
        )
        _, att_default = model_default(representations, position_mask)
        _, att_explicit = model_explicit(representations, position_mask)
        assert torch_module.equal(att_default, att_explicit)

    def test_omitted_and_explicit_produce_identical_predictions(
        self,
        shared_state,
        torch_module,
    ):
        """Default and explicit softmax models produce identical predictions."""
        state_dict, laplacian, representations, position_mask = shared_state
        model_default = self._build_model(torch_module, laplacian, state_dict)
        model_explicit = self._build_model(
            torch_module,
            laplacian,
            state_dict,
            attention_normalization="softmax",
        )
        pred_default, _ = model_default(representations, position_mask)
        pred_explicit, _ = model_explicit(representations, position_mask)
        assert torch_module.equal(pred_default, pred_explicit)

    def test_prediction_shape(self, shared_state, torch_module):
        """Both modes return predictions with shape [N]."""
        state_dict, laplacian, representations, position_mask = shared_state
        for kwargs in ({}, {"attention_normalization": "softmax"}):
            model = self._build_model(
                torch_module,
                laplacian,
                state_dict,
                **kwargs,
            )
            predictions, _ = model(representations, position_mask)
            assert predictions.shape == (self.N_LEAVES,)

    def test_attention_shape(self, shared_state, torch_module):
        """Both modes return attention with shape [N, L]."""
        state_dict, laplacian, representations, position_mask = shared_state
        for kwargs in ({}, {"attention_normalization": "softmax"}):
            model = self._build_model(
                torch_module,
                laplacian,
                state_dict,
                **kwargs,
            )
            _, attention = model(representations, position_mask)
            assert attention.shape == (self.N_LEAVES, self.N_POSITIONS)
