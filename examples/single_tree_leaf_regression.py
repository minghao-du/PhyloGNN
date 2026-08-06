"""Run deterministic leaf-level regression on a small in-memory tree."""

from ete3 import Tree
import torch

from phylognn import LeafRegressionConfig, run_leaf_regression
from phylognn.training import TrackingConfig

# Tracking is opt-in. Uncomment the next line and provide a W&B project to
# inspect fold and refit curves; the default remains local and credential-free.
# TRACKING_CONFIG: TrackingConfig | None = None
# TRACKING_CONFIG = TrackingConfig(enabled=True, project="your-project")
TRACKING_CONFIG = TrackingConfig(
      enabled=True,
      project="phylognn-leaf-regression-test",
      run_name="single-tree-leaf-regression",
      group="leaf-regression-manual-check",
      tags=("manual-check", "leaf-regression"),
  )


def main() -> None:
    """Print an in-memory leaf-regression workflow summary."""
    tree = Tree("((leaf_a:1,leaf_b:1):1,(leaf_c:1,(leaf_d:1,(leaf_e:1,leaf_f:1):1):1):1);")
    representations = torch.tensor(
        [
            [[0.2, 0.1, 0.5], [0.3, 0.4, 0.1], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.4, 0.2, 0.3], [0.1, 0.5, 0.2], [0.6, 0.1, 0.4], [0.0, 0.0, 0.0]],
            [[0.3, 0.6, 0.2], [0.2, 0.3, 0.5], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.5, 0.1, 0.4], [0.4, 0.3, 0.2], [0.1, 0.2, 0.6], [0.2, 0.5, 0.1]],
            [[0.1, 0.4, 0.6], [0.3, 0.2, 0.5], [0.5, 0.4, 0.1], [0.0, 0.0, 0.0]],
            [[0.6, 0.2, 0.1], [0.2, 0.6, 0.3], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    position_mask = torch.tensor(
        [
            [True, True, False, False],
            [True, True, True, False],
            [True, True, False, False],
            [True, True, True, True],
            [True, True, True, False],
            [True, True, False, False],
        ]
    )
    targets = torch.tensor([0.4, 1.2, -0.7, 2.1, 0.8, -1.5], dtype=torch.float32)
    result = run_leaf_regression(
        tree,
        representations,
        position_mask,
        targets,
        n_splits=3,
        training_config=LeafRegressionConfig(epochs=20, learning_rate=0.01, seed=7),
        tracking_config=TRACKING_CONFIG,
    )

    print("Single-tree leaf regression summary")
    print(f"leaf count: {len(tree)}")
    print(f"fold scores: {[round(score, 4) for score in result.fold_scores]}")
    print(f"overall score: {result.cv_score:.4f}")
    print(f"OOF predictions shape: {tuple(result.oof_predictions.shape)}")
    print(f"final predictions: {[round(value, 4) for value in result.predictions.tolist()]}")
    if result.mean_attention is None:
        print("attention summary: unavailable")
    else:
        print(f"attention summary: maximum mean position {int(result.mean_attention.argmax())}")


if __name__ == "__main__":
    main()
