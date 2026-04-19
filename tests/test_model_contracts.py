"""Contract tests for core model modules."""

import pytest

from tests.support import require_modules


torch = pytest.importorskip("torch")
require_modules("torch_geometric", "torch_scatter")

from torch_geometric.data import Data

from phylognn.models.base import BaseGATNet, BasePhyloGNN
from phylognn.models.gat_lstm import GATBiLSTMNet
from phylognn.models.layers import GATBlock


class _DummyPhyloModel(BasePhyloGNN):
    """Small concrete model for base-class validation tests."""

    def forward(self, data):
        return data.x

    def get_embedding_dim(self) -> int:
        return 1


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

