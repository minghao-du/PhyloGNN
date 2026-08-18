"""Behavior tests for the one-hot phylogenetic leaf regressor."""

import torch


def _make_model(**overrides):
    from phylognn.models.one_hot_phylo import OneHotPhyloRegressor

    kwargs = {
        "input_dim": 7,
        "leaf_laplacian": torch.eye(3, dtype=torch.float32),
        "num_bins": 4,
        "hidden_dim": 8,
        "species_dim": 6,
        "phylogeny_hidden_dim": 5,
        "dropout_prob": 0.0,
    }
    kwargs.update(overrides)
    return OneHotPhyloRegressor(**kwargs)


def _inputs():
    torch.manual_seed(7)
    representations = torch.randn(3, 6, 7)
    mask = torch.tensor(
        [
            [True, True, True, True, True, True],
            [True, True, True, True, False, False],
            [True, True, False, False, False, False],
        ]
    )
    return representations, mask


def test_forward_returns_prediction_only_and_matches_two_stage_api():
    """The generic leaf-regression fitter should receive one prediction per leaf."""
    model = _make_model().eval()
    representations, position_mask = _inputs()

    predictions = model(representations, position_mask)
    embeddings = model.encode_sequences(representations, position_mask)
    expected = model.predict_from_embeddings(embeddings)

    assert predictions.shape == (3,)
    assert torch.isfinite(predictions).all()
    assert torch.equal(predictions, expected)


def test_one_hot_pgls_composition_supports_forward_and_backward():
    """The ordered species embeddings compose with a differentiable PGLS head."""
    from phylognn.models import PGLSRegressionHead
    from phylognn.training import PGLSLoss

    model = _make_model().train()
    head = PGLSRegressionHead(6, 2)
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

    assert features.shape == (3, 6)
    assert predictions.shape == (3, 2)
    assert torch.isfinite(loss)
    assert model.species_encoder[0].weight.grad is not None
    assert head.linear.weight.grad is not None


def test_padding_values_and_padding_width_do_not_change_predictions():
    """Only valid positions and effective lengths should affect the model."""
    model = _make_model().eval()
    representations, position_mask = _inputs()
    representations[~position_mask] = 1000.0

    widened = torch.full((3, 8, 7), -500.0)
    widened[:, :6] = representations
    widened_mask = torch.cat(
        [position_mask, torch.zeros((3, 2), dtype=torch.bool)],
        dim=1,
    )

    expected = model(representations, position_mask)
    actual = model(widened, widened_mask)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=0)


def test_phylogeny_gate_starts_at_documented_value():
    """The phylogenetic residual starts small so sequence features lead training."""
    model = _make_model(phylogeny_gate_init=0.05)

    assert torch.allclose(
        torch.sigmoid(model.raw_phylogeny_gate), torch.tensor(0.05, dtype=torch.float32)
    )
    assert model.leaf_laplacian.requires_grad is False
