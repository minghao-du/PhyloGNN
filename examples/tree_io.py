"""Optional tree I/O example using repository sample data.

This example demonstrates the curated `phylognn.io` boundary for reading a tree
from `examples_data/simulated_trees/`. It is intentionally small: load one
tree, show a few high-signal statistics, and demonstrate that the resulting
`ete3.Tree` can feed into the core preprocessing workflow.

If the optional `dendropy` dependency is unavailable, the script prints concise
installation guidance and exits cleanly without a traceback.
"""

from pathlib import Path

from phylognn.data import TreeFeatureEngineer
from phylognn.io import read_tree_as_ete3

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_TREE = ROOT / "examples_data" / "simulated_trees" / "1.trees"


def main() -> None:
    try:
        tree = read_tree_as_ete3(SAMPLE_TREE)
    except (ModuleNotFoundError, ImportError, RuntimeError) as exc:
        message = str(exc)
        if "dendropy" not in message.lower():
            raise

        print("Optional dependency missing: dendropy")
        print('Install it with `python -m pip install -e ".[beast]"`.')
        return

    engineer = TreeFeatureEngineer(num_time_bins=8)
    featured_tree = engineer.add_features(
        tree,
        origin_time=8.0,
        feature_names=["node_time", "time_bin", "is_tip"],
        rescale=False,
        inplace=False,
    )

    tip_count = sum(1 for node in featured_tree.traverse() if node.is_leaf())
    first_tip = next((node.name for node in featured_tree.traverse() if node.is_leaf()), "unknown")

    print("Tree I/O summary")
    print(f"Loaded tree file: {SAMPLE_TREE.relative_to(ROOT)}")
    print(f"tip_count: {tip_count}")
    print(f"first_tip: {first_tip}")
    print(
        "next_step_features: "
        f"node_time={featured_tree.node_time:.2f}, time_bin={int(featured_tree.time_bin)}"
    )


if __name__ == "__main__":
    main()
