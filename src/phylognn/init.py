"""
PhyloGNN - Phylogenetic Tree to Graph Neural Network Data Conversion

A Python package for converting phylogenetic trees into graph neural network
compatible data structures using a flexible two-step pipeline.

Main Components:
- TreeFeatureEngineer: Add features/attributes to tree nodes
- TreeToGraphConverter: Convert tree structure to graph data

Quick Start:
>>> from ete3 import Tree
>>> from phylognn import TreeFeatureEngineer, TreeToGraphConverter
>>>
>>> # Step 1: Add features to tree
>>> engineer = TreeFeatureEngineer(num_time_bins=101)
>>> tree = Tree("((A:1,B:2)C:3,D:4)E;")
>>> tree_with_features = engineer.add_features(tree, origin_time=10.0)
>>>
>>> # Step 2: Convert to graph
>>> converter = TreeToGraphConverter(
...     feature_names=engineer.feature_names,
...     num_time_bins=101
... )
>>> data = converter.convert(tree_with_features)

For more examples, see the examples/ directory.
"""
from .data import TreeFeatureEngineer, TreeToGraphConverter

version = "0.1.0"
all = ['TreeFeatureEngineer', 'TreeToGraphConverter']

