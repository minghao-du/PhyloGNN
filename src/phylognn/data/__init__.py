"""
Phylogenetic Tree to Graph Data Conversion

This module provides a two-step pipeline for converting phylogenetic trees
into graph neural network compatible data structures:
    1. TreeFeatureEngineer: Adds features/attributes to tree nodes
    2. TreeToGraphConverter: Converts tree structure to graph data

This separation allows for flexible feature engineering and reusable conversion.

Example (recommended workflow):
from ete3 import Tree
from phylognn.data import TreeFeatureEngineer, TreeToGraphConverter

# Step 1: Add features to tree
engineer = TreeFeatureEngineer(
    num_time_bins=101,
    extant_sampling_probability=0.8
)
tree = Tree("((A:1,B:2)C:3,D:4)E;")
tree_with_features = engineer.add_features(tree, origin_time=10.0)

# Step 2: Convert to graph
converter = TreeToGraphConverter(
    feature_names=engineer.feature_names,
    add_virtual_nodes=True,
    num_time_bins=101
)
data = converter.convert(tree_with_features)
Example (custom features):
# Add your own features
def add_custom_feature(node, root, origin_time):
node.add_feature('my_feature', some_computation(node))

engineer = TreeFeatureEngineer(custom_features=[add_custom_feature])
tree_with_features = engineer.add_features(tree, origin_time=10.0)

# Convert with custom feature names
converter = TreeToGraphConverter(
    feature_names=['node_time', 'my_feature', 'is_tip']
)
data = converter.convert(tree_with_features)
"""

from .feature_engineer import TreeFeatureEngineer
from .converter import TreeToGraphConverter
from .tree_io import read_tree_as_ete3

__all__ = ['TreeFeatureEngineer', 'TreeToGraphConverter', 'read_tree_as_ete3']
