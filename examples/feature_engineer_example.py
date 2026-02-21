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

    print("\n")

def example_selective_features():
    """Add only specific features"""
    print("=" * 60)
    print("Example 2: Selective Features")
    print("=" * 60)
    
    tree = Tree("((A:1,B:2)C:3,D:4)E:0;", format=1)
    engineer = TreeFeatureEngineer(num_time_bins=50)
    
    # Add only specific features
    feature_names = ['node_time', 'is_tip', 'time_bin']
    tree_with_features = engineer.add_features(
        tree,
        origin_time=10.0,
        feature_names=feature_names,
        rescale=False
    )
    
    print(f"Added features: {feature_names}")
    print("\nNode features:")
    for node in tree_with_features.traverse():
        node_name = node.name if node.name else "internal"
        print(f"{node_name}: time={node.node_time:.4f}, "
              f"bin={node.time_bin}, is_tip={node.is_tip}")
    
    print("\n")

def example_rescaling():
    """Demonstrate tree rescaling"""
    print("=" * 60)
    print("Example 3: Tree Rescaling")
    print("=" * 60)
    
    tree = Tree("((A:1,B:2)C:3,D:4)E:0;", format=1)
    engineer = TreeFeatureEngineer()
    
    print("Original branch lengths:")
    for node in tree.traverse():
        node_name = node.name if node.name else "internal"
        print(f"  {node_name}: {node.dist:.4f}")
    
    # Rescale tree
    rescaled_tree, scale_factor, new_origin = engineer.rescale_tree(
        tree, 
        origin_time=10.0,
        inplace=False
    )
    
    print(f"\nScale factor: {scale_factor:.4f}")
    print(f"Original origin time: 10.0")
    print(f"New origin time: {new_origin:.4f}")
    
    print("\nRescaled branch lengths:")
    for node in rescaled_tree.traverse():
        node_name = node.name if node.name else "internal"
        print(f"  {node_name}: {node.dist:.4f}")
    
    print("\n")

def main():
    """Run all examples"""
    examples = [
        example_basic_usage,
        example_selective_features,
        example_rescaling,
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"Error in {example.__name__}: {e}\n")


if __name__ == "__main__":
    main()