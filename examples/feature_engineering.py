"""Self-contained TreeFeatureEngineer example."""

import sys
from pathlib import Path

from ete3 import Tree

# Make the local `src/` package importable when running the script directly.
ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from phylognn.data import TreeFeatureEngineer


FEATURE_NAMES = [
    "node_time",
    "time_bin",
    "branch_length",
    "is_tip",
    "is_internal",
]


def build_demo_tree() -> Tree:
    return Tree("((A:1.0,B:1.5)C:0.5,D:2.0)root:0.0;", format=1)


def main() -> None:
    engineer = TreeFeatureEngineer(num_time_bins=6)
    tree = engineer.add_features(
        build_demo_tree(),
        origin_time=4.0,
        feature_names=FEATURE_NAMES,
        rescale=False,
        inplace=True,
    )

    print("Feature engineering summary")
    print(f"Feature order: {FEATURE_NAMES}")
    for node in tree.traverse(engineer.traversal_strategy):
        label = node.name or "internal"
        print(
            f"{label}: "
            f"node_time={node.node_time:.2f}, "
            f"time_bin={int(node.time_bin)}, "
            f"branch_length={node.branch_length:.2f}, "
            f"is_tip={int(node.is_tip)}, "
            f"is_internal={int(node.is_internal)}"
        )


if __name__ == "__main__":
    main()
