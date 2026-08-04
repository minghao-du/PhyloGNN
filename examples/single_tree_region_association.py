"""Evaluate one leaf-aligned region against a small in-memory tree."""

from ete3 import Tree
import torch

from phylognn import evaluate_region_association


def main() -> None:
    """Run a deterministic, no-persistence region association workflow."""
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

    result = evaluate_region_association(
        tree,
        representations,
        position_mask,
        targets,
        n_splits=3,
        epochs=20,
        hidden_dim=8,
        learning_rate=0.01,
        seed=7,
    )

    print("Single-tree region association summary")
    print(f"leaf count: {len(tree)}")
    print(f"representations shape: {tuple(representations.shape)}")
    print(f"position mask shape: {tuple(position_mask.shape)}")
    print(f"fold R2: {[round(score, 4) for score in result.fold_r2]}")
    print(f"cv R2: {result.cv_r2:.4f}")
    print(f"maximum mean-attention position: {int(result.mean_attention.argmax())}")


if __name__ == "__main__":
    main()
