"""
Examples for TreeFeatureEngineer

This module demonstrates various use cases of the TreeFeatureEngineer class
for adding features to phylogenetic trees.
"""

from ete3 import Tree
from phylognn.data import TreeFeatureEngineer


def _node_label(node):
    """Return a readable label for a node."""
    return node.name if node.name else "internal"


def _print_basic_node_info(tree, engineer, show_features):
    """Print selected features for all nodes in a tree."""
    for node in tree.traverse(engineer.traversal_strategy):
        node_name = _node_label(node)
        values = []
        for feature_name in show_features:
            value = getattr(node, feature_name)
            if isinstance(value, float):
                values.append(f"{feature_name}={value:.4f}")
            else:
                values.append(f"{feature_name}={value}")
        print(f"  {node_name}: " + ", ".join(values))


def example_basic_usage():
    """Basic usage: add all features to a simple tree."""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)

    tree = Tree("((A:1,B:2)C:3,D:4)E:0;", format=1)
    print(f"Original tree: {tree.write(format=1)}")

    engineer = TreeFeatureEngineer(num_time_bins=10)
    print(f"Registered features: {engineer.feature_names}")

    tree_with_features = engineer.add_features(
        tree,
        origin_time=10.0,
        feature_names=None,  # add all features
        rescale=True,
        inplace=True,
    )

    print("\nNode features:")
    _print_basic_node_info(
        tree_with_features,
        engineer,
        show_features=[
            "node_time",
            "time_bin",
            "is_tip",
            "is_internal",
            "branch_length",
        ],
    )
    print()


def example_selective_features():
    """Add only specific features."""
    print("=" * 60)
    print("Example 2: Selective Features")
    print("=" * 60)

    tree = Tree("((A:1,B:2)C:3,D:4)E:0;", format=1)
    engineer = TreeFeatureEngineer(num_time_bins=50)

    feature_names = ["node_time", "is_tip", "time_bin"]
    tree_with_features = engineer.add_features(
        tree,
        origin_time=10.0,
        feature_names=feature_names,
        rescale=False,
        inplace=True,
    )

    print(f"Requested features: {feature_names}")
    print("\nNode features:")
    _print_basic_node_info(
        tree_with_features,
        engineer,
        show_features=feature_names,
    )
    print()


def example_rescaling():
    """Demonstrate tree rescaling."""
    print("=" * 60)
    print("Example 3: Tree Rescaling")
    print("=" * 60)

    tree = Tree("((A:1,B:2)C:3,D:4)E:0;", format=1)
    engineer = TreeFeatureEngineer()

    print("Original branch lengths:")
    for node in tree.traverse(engineer.traversal_strategy):
        print(f"  {_node_label(node)}: {node.dist:.4f}")

    rescaled_tree, scale_factor, new_origin = engineer.rescale_tree(
        tree,
        origin_time=10.0,
        inplace=False,
    )

    print(f"\nScale factor: {scale_factor:.4f}")
    print("Original origin time: 10.0000")
    print(f"New origin time: {new_origin:.4f}")

    print("\nRescaled branch lengths:")
    for node in rescaled_tree.traverse(engineer.traversal_strategy):
        print(f"  {_node_label(node)}: {node.dist:.4f}")

    print("\nOriginal tree remains unchanged because inplace=False:")
    for node in tree.traverse(engineer.traversal_strategy):
        print(f"  {_node_label(node)}: {node.dist:.4f}")

    print()


def example_custom_feature():
    """Add a custom feature to each node."""
    print("=" * 60)
    print("Example 4: Custom Feature")
    print("=" * 60)

    def add_name_length(context):
        node = context["node"]
        node.add_feature("name_length", len(node.name) if node.name else 0)

    tree = Tree("((AA:1,B:2)C:3,D:4)E:0;", format=1)

    engineer = TreeFeatureEngineer(
        num_time_bins=10,
        custom_features={"name_length": add_name_length},
    )

    tree_with_features = engineer.add_features(
        tree,
        origin_time=10.0,
        feature_names=["node_time", "name_length", "is_tip"],
        rescale=False,
        inplace=True,
    )

    print("Node features:")
    _print_basic_node_info(
        tree_with_features,
        engineer,
        show_features=["node_time", "name_length", "is_tip"],
    )
    print()


def main():
    """Run all examples."""
    examples = [
        example_basic_usage,
        example_selective_features,
        example_rescaling,
        example_custom_feature,
    ]

    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"Error in {example.__name__}: {e}\n")


if __name__ == "__main__":
    main()
