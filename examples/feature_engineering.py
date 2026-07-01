"""Self-contained TreeFeatureEngineer example."""

from ete3 import Tree

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

    # --- Per-tree extant_sampling_probability ---
    print("\n--- Per-tree extant_sampling_probability ---")
    tree_a = build_demo_tree()
    tree_b = build_demo_tree()

    tree_a = engineer.add_features(
        tree_a,
        origin_time=4.0,
        extant_sampling_probability=0.8,
        rescale=False,
    )
    tree_b = engineer.add_features(
        tree_b,
        origin_time=4.0,
        extant_sampling_probability=0.5,
        rescale=False,
    )

    for label, t, expected in [("Tree A", tree_a, 0.8), ("Tree B", tree_b, 0.5)]:
        root = t.get_tree_root()
        print(f"{label}: extant_sampling_probability = {root.extant_sampling_probability}")


if __name__ == "__main__":
    main()
