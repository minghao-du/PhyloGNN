"""Contract tests for core model modules."""

import pytest

from tests.support import require_modules


torch = pytest.importorskip("torch")
nn = torch.nn
require_modules("torch_geometric", "torch_scatter")

from torch_geometric.data import Data  # noqa: E402

from phylognn.models.base import BaseGATNet, BasePhyloGNN  # noqa: E402
from phylognn.models.gat_lstm import GATBiLSTMNet  # noqa: E402
from phylognn.models.layers import GATBlock, TemporalBiLSTMEncoder  # noqa: E402


class _DummyPhyloModel(BasePhyloGNN):
    """Small concrete model for base-class validation tests."""

    def forward(self, data):
        return data.x

    def get_embedding_dim(self) -> int:
        return 1


class _FixedSequenceModule(nn.Module):
    """Return a fixed recurrent output for aggregation-focused tests."""

    def __init__(self, recurrent_output):
        super().__init__()
        self.recurrent_output = recurrent_output
        self.calls = 0

    def forward(self, sequence):
        self.calls += 1
        return self.recurrent_output.to(sequence.device, sequence.dtype), None


def _tiny_temporal_graph():
    return Data(
        x=torch.tensor(
            [
                [1.0, 0.0, 0.5, 1.0],
                [0.5, 1.0, 0.0, 1.0],
                [1.0, 1.0, 0.5, 0.0],
                [0.0, 1.0, 1.0, 0.5],
            ],
            dtype=torch.float,
        ),
        edge_index=torch.tensor(
            [
                [0, 1, 2, 3, 0, 2],
                [1, 0, 3, 2, 2, 0],
            ],
            dtype=torch.long,
        ),
        batch=torch.tensor([0, 0, 1, 1], dtype=torch.long),
        time_bin=torch.tensor([0, 1, 0, 1], dtype=torch.long),
    )


def test_validate_data_rejects_non_floating_x():
    """Base validation should reject integer feature tensors early."""
    model = _DummyPhyloModel()
    data = Data(
        x=torch.ones((3, 2), dtype=torch.long),
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
    )

    with pytest.raises(TypeError, match="floating-point"):
        model.validate_data(data)


def test_base_gat_net_rejects_invalid_encoder_type():
    """Encoder validation should fail before any module is built."""
    with pytest.raises(ValueError, match="encoder_type"):
        BaseGATNet._validate_init_args(
            input_dim=4,
            preprocess_dim=8,
            gat_hidden_dim=16,
            gat_heads=2,
            num_gat_layers=2,
            dropout_prob=0.1,
            use_preprocessing=True,
            encoder_type="bad",
        )


def test_gat_block_rejects_invalid_head_count():
    """Low-level layers should preserve their explicit constructor contract."""
    with pytest.raises(ValueError, match="heads"):
        GATBlock(in_channels=4, out_channels=8, heads=0)


def test_gat_bilstm_requires_num_time_bins_for_temporal_modes():
    """Temporal models must require explicit time-bin configuration."""
    with pytest.raises(ValueError, match="num_time_bins"):
        GATBiLSTMNet(input_dim=4, output_dim=1, temporal_mode="fc", num_time_bins=None)


def test_temporal_bilstm_encoder_rejects_invalid_constructor_args():
    """The reusable temporal encoder should fail fast on bad configuration."""
    with pytest.raises(ValueError, match="input_dim"):
        TemporalBiLSTMEncoder(input_dim=0, hidden_dim=4)

    with pytest.raises(ValueError, match="hidden_dim"):
        TemporalBiLSTMEncoder(input_dim=3, hidden_dim=0)

    with pytest.raises(ValueError, match="num_layers"):
        TemporalBiLSTMEncoder(input_dim=3, hidden_dim=4, num_layers=0)

    with pytest.raises(ValueError, match="dropout_prob"):
        TemporalBiLSTMEncoder(input_dim=3, hidden_dim=4, dropout_prob=1.0)

    with pytest.raises(ValueError, match="aggregation"):
        TemporalBiLSTMEncoder(input_dim=3, hidden_dim=4, aggregation="sum")


def test_temporal_bilstm_encoder_forward_shape_and_input_validation():
    """Sequence inputs must be rank-3 with the configured feature dimension."""
    encoder = TemporalBiLSTMEncoder(input_dim=3, hidden_dim=5, dropout_prob=0.0)
    sequence = torch.randn(2, 4, 3)

    out = encoder(sequence)

    assert out.shape == (2, 10)

    with pytest.raises(ValueError, match="3D"):
        encoder(torch.randn(4, 3))

    with pytest.raises(ValueError, match="feature dim"):
        encoder(torch.randn(2, 4, 2))


@pytest.mark.parametrize(
    ("aggregation", "expected"),
    [
        ("mean", [[3.0, 4.0], [9.0, 10.0]]),
        ("last", [[5.0, 6.0], [11.0, 12.0]]),
        ("max", [[5.0, 6.0], [11.0, 12.0]]),
    ],
)
def test_temporal_bilstm_encoder_aggregation_rules(aggregation, expected):
    """Mean, last, and max aggregation should operate on one recurrent pass."""
    recurrent_output = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
        ]
    )
    encoder = TemporalBiLSTMEncoder(
        input_dim=2,
        hidden_dim=1,
        dropout_prob=0.0,
        aggregation=aggregation,
    )
    fixed_lstm = _FixedSequenceModule(recurrent_output)
    encoder.input_norm = nn.Identity()
    encoder.lstm = fixed_lstm
    encoder.output_norm = nn.Identity()
    encoder.dropout = nn.Identity()

    out = encoder(torch.randn(2, 3, 2))

    assert torch.allclose(out, torch.tensor(expected))
    assert fixed_lstm.calls == 1


def test_temporal_bilstm_encoder_reset_parameters_changes_weights():
    """The reusable temporal encoder should expose explicit reset behavior."""
    encoder = TemporalBiLSTMEncoder(input_dim=3, hidden_dim=4, dropout_prob=0.0)
    before = encoder.lstm.weight_ih_l0.detach().clone()

    encoder.reset_parameters()

    assert not torch.equal(before, encoder.lstm.weight_ih_l0)


def test_temporal_bilstm_encoder_rejects_graph_like_tensor_boundary():
    """The generic encoder consumes prepared sequences, not node-level graphs."""
    encoder = TemporalBiLSTMEncoder(input_dim=3, hidden_dim=4, dropout_prob=0.0)

    with pytest.raises(ValueError, match="3D"):
        encoder(torch.randn(5, 3))


def test_gat_bilstm_uses_reusable_temporal_encoder_in_lstm_mode():
    """LSTM temporal mode should compose the reusable encoder module."""
    model = GATBiLSTMNet(
        input_dim=4,
        output_dim=1,
        temporal_mode="lstm",
        num_time_bins=2,
        temporal_hidden_dim=8,
    )

    assert isinstance(model.temporal_encoder, TemporalBiLSTMEncoder)
    assert model.temporal_encoder.aggregation == "mean"


def test_gat_bilstm_accepts_default_temporal_aggregation():
    """The default aggregation should preserve current LSTM-mode behavior."""
    model = GATBiLSTMNet(
        input_dim=4,
        output_dim=1,
        temporal_mode="lstm",
        num_time_bins=2,
    )

    assert model.temporal_aggregation == "mean"


def test_gat_bilstm_accepts_non_default_temporal_aggregation():
    """Users should be able to select a supported recurrent aggregation."""
    model = GATBiLSTMNet(
        input_dim=4,
        output_dim=1,
        temporal_mode="lstm",
        num_time_bins=2,
        temporal_aggregation="last",
    )

    assert model.temporal_encoder.aggregation == "last"


def test_gat_bilstm_rejects_invalid_temporal_aggregation():
    """Unsupported recurrent aggregation names should fail during construction."""
    with pytest.raises(ValueError, match="temporal_aggregation"):
        GATBiLSTMNet(
            input_dim=4,
            output_dim=1,
            temporal_mode="lstm",
            num_time_bins=2,
            temporal_aggregation="sum",
        )


def test_gat_bilstm_non_lstm_modes_do_not_create_recurrent_encoder():
    """Non-temporal and FC temporal paths should not depend on the LSTM encoder."""
    none_model = GATBiLSTMNet(input_dim=4, output_dim=1, temporal_mode="none")
    fc_model = GATBiLSTMNet(input_dim=4, output_dim=1, temporal_mode="fc", num_time_bins=2)

    assert not hasattr(none_model, "temporal_encoder")
    assert not hasattr(fc_model, "temporal_encoder")


def test_gat_bilstm_exposes_named_temporal_dimensions_in_lstm_mode():
    """Temporal dimensions should be inspectable without magic-number arithmetic."""
    model = GATBiLSTMNet(
        input_dim=4,
        output_dim=1,
        temporal_mode="lstm",
        num_time_bins=2,
        gat_hidden_dim=3,
        gat_heads=2,
        temporal_hidden_dim=5,
    )

    assert model.time_bin_scalar_feature_dim == 1
    assert model.temporal_input_dim == model.get_embedding_dim() + model.time_bin_scalar_feature_dim
    assert model.temporal_output_dim == model.temporal_encoder.output_dim == 10


def test_gat_bilstm_default_fc_hidden_dims_preserve_existing_widths():
    """Omitting FC widths should preserve the historical hidden-size-derived widths."""
    model = GATBiLSTMNet(
        input_dim=4,
        output_dim=1,
        temporal_mode="fc",
        num_time_bins=3,
        gat_hidden_dim=2,
        gat_heads=1,
        temporal_hidden_dim=5,
    )

    linear_layers = [module for module in model.temporal_mlp if isinstance(module, nn.Linear)]

    assert model.time_bin_scalar_feature_dim == 1
    assert model.temporal_input_dim == 3
    assert model.resolved_temporal_fc_hidden_dims == (20, 10)
    assert model.temporal_output_dim == 10
    assert [(layer.in_features, layer.out_features) for layer in linear_layers] == [
        (9, 20),
        (20, 10),
    ]


def test_gat_bilstm_accepts_custom_fc_hidden_dims_with_variable_depth():
    """Explicit FC widths should define both temporal MLP depth and output width."""
    model = GATBiLSTMNet(
        input_dim=4,
        output_dim=1,
        temporal_mode="fc",
        num_time_bins=2,
        gat_hidden_dim=2,
        gat_heads=1,
        temporal_hidden_dim=5,
        temporal_fc_hidden_dims=(7, 5, 3),
    )

    linear_layers = [module for module in model.temporal_mlp if isinstance(module, nn.Linear)]

    assert model.resolved_temporal_fc_hidden_dims == (7, 5, 3)
    assert model.temporal_output_dim == 3
    assert [(layer.in_features, layer.out_features) for layer in linear_layers] == [
        (6, 7),
        (7, 5),
        (5, 3),
    ]
    assert "temporal_fc_hidden_dims=(7, 5, 3)" in model.extra_repr()


@pytest.mark.parametrize("temporal_fc_hidden_dims", [(), (8, 0), (-1,), (8.0,), ("8",)])
def test_gat_bilstm_rejects_invalid_fc_hidden_dims(temporal_fc_hidden_dims):
    """Explicit FC width sequences must be non-empty positive integer sequences."""
    with pytest.raises((TypeError, ValueError), match="temporal_fc_hidden_dims"):
        GATBiLSTMNet(
            input_dim=4,
            output_dim=1,
            temporal_mode="fc",
            num_time_bins=2,
            temporal_fc_hidden_dims=temporal_fc_hidden_dims,
        )


def test_gat_bilstm_lstm_forward_shape_is_unchanged():
    """A valid batched temporal graph should still produce graph-level outputs."""
    model = GATBiLSTMNet(
        input_dim=4,
        output_dim=2,
        temporal_mode="lstm",
        num_time_bins=2,
        gat_hidden_dim=2,
        gat_heads=1,
        num_gat_layers=1,
        temporal_hidden_dim=3,
        dropout_prob=0.0,
    )
    model.eval()

    out = model(_tiny_temporal_graph())

    assert out.shape == (2, 2)


def test_gat_bilstm_keeps_graph_specific_time_bin_pooling_boundary():
    """Graph-specific time-bin pooling should remain on the graph-temporal model."""
    pooled = GATBiLSTMNet._pool_by_time_bins(
        x=torch.tensor([[1.0, 0.0], [2.0, 0.0], [0.0, 3.0]]),
        time_bin=torch.tensor([0, 1, 0], dtype=torch.long),
        batch=torch.tensor([0, 0, 1], dtype=torch.long),
        num_time_bins=2,
    )

    assert pooled.shape == (2, 2, 3)
    assert pooled.size(-1) == 2 + GATBiLSTMNet.TIME_BIN_SCALAR_FEATURE_DIM
    assert torch.allclose(pooled[0, :, -1], torch.tensor([0.0, 1.0]))
