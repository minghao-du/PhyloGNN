"""Tests for the GATNodeRegressor model and node-level prediction contracts."""

import pytest

from tests.support import require_modules

torch = pytest.importorskip("torch")
nn = torch.nn
require_modules("torch_geometric", "torch_scatter")

from ete3 import Tree  # noqa: E402
from torch_geometric.data import Data  # noqa: E402

import examples.extant_trait_regression as extant_example  # noqa: E402
from examples.extant_trait_regression import (  # noqa: E402
    MaskedNodeRegressor,
    build_graph,
    create_masked_graph_view,
    create_masks,
    create_training_config,
    validate_graph_data,
)
from phylognn.models.gat_node import GATNodeRegressor  # noqa: E402
from phylognn.training.config import SUPPORTED_MODEL_TYPES  # noqa: E402


def _node_graph(num_nodes: int = 10, num_features: int = 3) -> Data:
    """Create a simple test graph with the given number of nodes and features."""
    x = torch.randn(num_nodes, num_features)
    # Simple chain graph: 0->1->2->...->N-1->0
    src = list(range(num_nodes))
    dst = [(i + 1) % num_nodes for i in range(num_nodes)]
    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
    return Data(x=x, edge_index=edge_index)


class TestGATNodeRegressorShape:
    """Tests for GATNodeRegressor output shape contracts."""

    def test_output_shape_matches_num_nodes(self):
        """Node-level model should produce one prediction per node."""
        data = _node_graph(num_nodes=10, num_features=3)
        model = GATNodeRegressor(input_dim=3, output_dim=1, preprocess_dim=16, gat_hidden_dim=16)
        model.eval()

        out = model(data)

        assert out.shape == (10, 1)

    def test_output_dim_configurable(self):
        """Output dimension should be configurable to multi-dimensional outputs."""
        data = _node_graph(num_nodes=10, num_features=3)
        model = GATNodeRegressor(input_dim=3, output_dim=3, preprocess_dim=16, gat_hidden_dim=16)
        model.eval()

        out = model(data)

        assert out.shape == (10, 3)

    def test_forward_no_batch_required(self):
        """Node-level model should work without a batch attribute on Data."""
        data = _node_graph(num_nodes=5, num_features=4)
        assert not hasattr(data, "batch") or data.batch is None
        model = GATNodeRegressor(input_dim=4, output_dim=1, preprocess_dim=8, gat_hidden_dim=8)
        model.eval()

        out = model(data)

        assert out.shape == (5, 1)


class TestGATNodeRegressorMasks:
    """Tests for mask application and lookup logic in node-level prediction."""

    def test_mask_applied_to_loss(self):
        """Loss computed with NaN masking should exclude NaN nodes."""
        data = _node_graph(num_nodes=10, num_features=3)
        model = GATNodeRegressor(input_dim=3, output_dim=1, preprocess_dim=16, gat_hidden_dim=16)
        model.eval()

        # Create target with some NaN values
        y = torch.randn(10)
        y[3] = float("nan")
        y[7] = float("nan")

        out = model(data).squeeze(-1)
        prediction_mask = ~torch.isnan(y)

        # Loss should be computable and finite with masked values
        masked_loss = nn.MSELoss()(out[prediction_mask], y[prediction_mask])

        assert torch.isfinite(masked_loss)
        assert prediction_mask.sum() == 8

    def test_train_val_test_mask_split(self):
        """Train, val, and test masks should be mutually exclusive and cover valid nodes."""
        num_nodes = 20
        # Simulate prediction_mask: first 15 are valid, last 5 are NaN
        prediction_mask = torch.zeros(num_nodes, dtype=torch.bool)
        prediction_mask[:15] = True

        valid_indices = prediction_mask.nonzero(as_tuple=True)[0]
        n_valid = len(valid_indices)
        perm = torch.randperm(n_valid)

        n_train = int(0.7 * n_valid)
        n_val = int(0.2 * n_valid)

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[valid_indices[perm[:n_train]]] = True
        val_mask[valid_indices[perm[n_train : n_train + n_val]]] = True
        test_mask[valid_indices[perm[n_train + n_val :]]] = True

        # Mutual exclusivity
        assert (train_mask & val_mask).sum() == 0
        assert (train_mask & test_mask).sum() == 0
        assert (val_mask & test_mask).sum() == 0

        # Union equals prediction_mask
        assert (train_mask | val_mask | test_mask).equal(prediction_mask)

    def test_missing_mask_raises_attribute_error(self):
        """get_mask helper should raise AttributeError when mask is absent."""

        def get_mask(data: Data, mask_key: str | None) -> torch.Tensor:
            """Replicate the example script's mask lookup logic."""
            if mask_key is None:
                if hasattr(data, "prediction_mask") and data.prediction_mask is not None:
                    return data.prediction_mask
                raise AttributeError(
                    "Data object has no 'prediction_mask' attribute. "
                    "Ensure prediction_mask is set before training."
                )
            if hasattr(data, mask_key):
                mask = getattr(data, mask_key)
                if mask is not None:
                    return mask
            if hasattr(data, "prediction_mask") and data.prediction_mask is not None:
                return data.prediction_mask
            raise AttributeError(
                f"Data object has no '{mask_key}' or 'prediction_mask' attribute. "
                "Ensure masks are set before training."
            )

        data = Data(
            x=torch.randn(5, 3),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        )

        with pytest.raises(AttributeError, match="prediction_mask"):
            get_mask(data, None)

        with pytest.raises(AttributeError, match="train_mask"):
            get_mask(data, "train_mask")


class TestMetricPostprocess:
    """Tests for metric postprocessing in node-level regression."""

    def test_postprocess_fn_applied_before_metrics(self):
        """Applying expm1 as postprocess before MSE should match manual calculation."""
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.1, 2.2, 2.8])

        postprocess_fn = torch.expm1
        pred_post = postprocess_fn(pred)
        target_post = postprocess_fn(target)

        mse = nn.MSELoss()(pred_post, target_post)
        manual_mse = ((pred_post - target_post) ** 2).mean()

        assert torch.allclose(mse, manual_mse)


class TestGATNodeRegressorConfig:
    """Tests for GATNodeRegressor registration in training config."""

    def test_registered_in_supported_model_types(self):
        """GATNodeRegressor should be accepted by the TOML config system."""
        assert "GATNodeRegressor" in SUPPORTED_MODEL_TYPES


class TestEdgeCases:
    """Tests for edge cases in node-level regression."""

    def test_zero_valid_prediction_nodes_raises_error(self):
        """When all y values are NaN, create_masks should raise ValueError."""
        num_nodes = 10
        y = torch.full((num_nodes,), float("nan"))
        prediction_mask = ~torch.isnan(y)

        assert prediction_mask.sum() == 0

        with pytest.raises(ValueError, match="zero"):
            create_masks(prediction_mask)

    def test_missing_leaf_labels_fail_before_training(self):
        """A tree with no valid extant labels fails during mask creation."""
        tree = Tree("(A:1.0,B:1.0)root:1.0;", format=1)
        with pytest.warns(UserWarning, match="no matching trait row"):
            data = build_graph(tree, {})

        with pytest.raises(ValueError, match="zero valid"):
            create_masks(data.prediction_mask)

    @pytest.mark.parametrize("size", [float("nan"), float("inf"), float("-inf")])
    def test_nonfinite_sizes_are_excluded_from_prediction_mask(self, size):
        tree = Tree("(A:1.0,B:1.0)root:1.0;", format=1)
        data = build_graph(
            tree,
            {"A": {"size": size, "range": 3.0}, "B": {"size": 2.0, "range": 3.0}},
        )

        index = data.node_names.index("A")
        assert not bool(data.prediction_mask[index])
        assert torch.isfinite(data.y[data.prediction_mask]).all()

    def test_unmatched_leaf_range_is_nan(self):
        """Missing leaf traits use NaN while internal range uses the sentinel."""
        tree = Tree("(A:1.0,B:1.0)root:1.0;", format=1)
        with pytest.warns(UserWarning, match="no matching trait row"):
            data = build_graph(tree, {"A": {"size": 2.0, "range": 3.0}})

        unmatched_index = data.node_names.index("B")
        root_index = data.node_names.index("root")
        assert torch.isnan(data.x[unmatched_index, 2])
        assert data.x[root_index, 2].item() == -1.0

    def test_constant_range_feature_training_smoke(self, tmp_path, monkeypatch):
        """The training loop should tolerate zero variance in the range feature."""
        data = Data(
            x=torch.ones(4, 3),
            edge_index=torch.tensor(
                [[0, 1, 2, 3], [1, 2, 3, 0]],
                dtype=torch.long,
            ),
            y=torch.tensor([1.0, 1.1, 0.9, 1.2]),
            train_mask=torch.tensor([True, True, False, False]),
            val_mask=torch.tensor([False, False, True, False]),
            test_mask=torch.tensor([False, False, False, True]),
            prediction_mask=torch.ones(4, dtype=torch.bool),
        )
        model = GATNodeRegressor(
            input_dim=3,
            output_dim=1,
            preprocess_dim=4,
            gat_hidden_dim=4,
            gat_heads=1,
            num_gat_layers=1,
            head_hidden_dim=4,
        )
        monkeypatch.setattr(extant_example, "OUTPUT_DIR", str(tmp_path))

        history = extant_example.train_model(model, data, epochs=1, lr=1e-3)

        assert len(history["train_loss"]) == 1

    def test_metric_postprocess_callback_called_once_per_epoch(self, tmp_path, monkeypatch):
        """The callback receives complete masked tensors once per validation epoch."""
        data = Data(
            x=torch.ones(4, 3),
            edge_index=torch.tensor(
                [[0, 1, 2, 3], [1, 2, 3, 0]],
                dtype=torch.long,
            ),
            y=torch.tensor([1.0, 1.1, 0.9, 1.2]),
            train_mask=torch.tensor([True, True, False, False]),
            val_mask=torch.tensor([False, False, True, True]),
            prediction_mask=torch.ones(4, dtype=torch.bool),
        )
        model = GATNodeRegressor(
            input_dim=3,
            output_dim=1,
            preprocess_dim=4,
            gat_hidden_dim=4,
            gat_heads=1,
            num_gat_layers=1,
            head_hidden_dim=4,
        )
        monkeypatch.setattr(extant_example, "OUTPUT_DIR", str(tmp_path))
        calls: list[tuple[int, int]] = []

        def postprocess(
            predictions: torch.Tensor,
            targets: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            calls.append((predictions.numel(), targets.numel()))
            return predictions, targets

        extant_example.train_model(
            model,
            data,
            epochs=2,
            lr=1e-3,
            metric_postprocess_fn=postprocess,
        )

        assert calls == [(2, 2), (2, 2)]


class TestTrainerIntegrationContracts:
    """Regression tests for the example's graph and Trainer adapters."""

    def _valid_data(self) -> Data:
        return Data(
            x=torch.ones(4, 3),
            edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long),
            y=torch.tensor([1.0, 1.1, 2.0, float("inf")]),
            prediction_mask=torch.tensor([True, True, True, False]),
            train_mask=torch.tensor([True, False, False, False]),
            val_mask=torch.tensor([False, True, False, False]),
            test_mask=torch.tensor([False, False, True, False]),
        )

    def test_validation_rejects_nonfinite_prediction_nodes_and_bad_splits(self):
        data = self._valid_data()
        data.prediction_mask = torch.tensor([True, True, True, True])
        with pytest.raises(ValueError, match="prediction_mask"):
            validate_graph_data(data)

        data = self._valid_data()
        data.test_mask = torch.tensor([False, False, False, False])
        with pytest.raises(ValueError, match="test_mask"):
            validate_graph_data(data)

    def test_masked_view_and_adapter_keep_full_graph_and_align_targets(self):
        data = Data(
            x=torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
            y=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        )
        node_mask = torch.tensor([True, False, True, False])
        view = create_masked_graph_view(data, node_mask)
        assert view.x.shape == data.x.shape
        assert torch.equal(view.edge_index, data.edge_index)
        assert view.y.shape == (2,)

        class BaseModel(nn.Module):
            def forward(self, graph: Data) -> torch.Tensor:
                return graph.x[:, :1]

        predictions = MaskedNodeRegressor(BaseModel())(view)
        assert predictions.shape == (2, 1)
        assert torch.equal(predictions[:, 0], view.y)

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"epochs": 0}, "epochs"),
            ({"learning_rate": 0.0}, "learning_rate"),
            ({"save_dir": ""}, "save_dir"),
        ],
    )
    def test_training_config_rejects_invalid_values(self, kwargs, message):
        with pytest.raises((TypeError, ValueError), match=message):
            create_training_config(**kwargs)
