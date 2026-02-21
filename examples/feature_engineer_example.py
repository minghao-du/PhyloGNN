"""
Examples for TreeFeatureEngineer

This module demonstrates various use cases of the TreeFeatureEngineer class
for adding features to phylogenetic trees.
"""

from ete3 import Tree
from phylognn.data import TreeFeatureEngineer

def example_basic_usage():
    """Basic usage: Add all features to a simple tree"""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    # Create a simple tree
    tree = Tree("((A:1,B:2)C:3,D:4)E:0;", format=1)
    print(f"Original tree: {tree.write()}")
    
    # Initialize feature engineer
    engineer = TreeFeatureEngineer(num_time_bins=10)
    
    # Add all features with rescaling
    tree_with_features = engineer.add_features(
        tree, 
        origin_time=10.0, 
        rescale=True
    )
    
    # Display features for each node
    print("\nNode features:")
    for node in tree_with_features.traverse():
        node_name = node.name if node.name else "internal"
        print(f"\n{node_name}:")
        print(f"  node_time: {node.node_time:.4f}")
        print(f"  time_bin: {node.time_bin}")
        print(f"  is_tip: {node.is_tip}")
        print(f"  is_internal: {node.is_internal}")
        print(f"  branch_length: {node.branch_length:.4f}")

def main():
    """Run all examples"""
    examples = [
        example_basic_usage,
        # example_selective_features,
        # example_rescaling,
        # example_fossil_detection,
        # example_sampled_ancestors,
        # example_time_bins,
        # example_custom_features,
        # example_sampling_probability,
        # example_inplace_vs_copy
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"Error in {example.__name__}: {e}\n")


if __name__ == "__main__":
    main()