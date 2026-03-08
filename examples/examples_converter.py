"""
Examples for TreeToGraphConverter

This module demonstrates various use cases of the TreeToGraphConverter class
for converting phylogenetic trees with node features into PyTorch Geometric
Data objects, including saving and loading graph data.
"""

from pathlib import Path

from ete3 import Tree
from phylognn.data import TreeFeatureEngineer, TreeToGraphConverter


def _safe_node_name(name):
    """Return a readable node name."""
    return name if name else "internal"


def _print_data_summary(data, converter):
    """Print a summary of a PyG Data object."""
    print("Graph summary:")
    print(f"  num_nodes: {data.num_nodes}")
    print(f"  num_edges: {data.edge_index.size(1)}")
    print(f"  x.shape: {tuple(data.x.shape)}")
    print(f"  feature_names: {converter.output_feature_names}")

    if hasattr(data, "original_num_nodes"):
        print(f"  original_num_nodes: {data.original_num_nodes}")

    if hasattr(data, "num_time_bins"):
        print(f"  num_time_bins: {data.num_time_bins}")

    if hasattr(data, "edge_type"):
        unique_edge_types = sorted(set(data.edge_type.tolist()))
        print(f"  edge_types_present: {unique_edge_types}")

    if hasattr(data, "node_names"):
        print(f"  node_names: {data.node_names}")


def _print_node_features(data, converter, max_nodes=None):
    """Print node feature rows."""
    print("Node features:")
    feature_names = converter.output_feature_names
    limit = data.num_nodes if max_nodes is None else min(max_nodes, data.num_nodes)

    for i in range(limit):
        row = []
        node_name = data.node_names[i] if hasattr(data, "node_names") else str(i)
        for j, feature_name in enumerate(feature_names):
            value = data.x[i, j].item()
            if float(value).is_integer():
                row.append(f"{feature_name}={int(value)}")
            else:
                row.append(f"{feature_name}={value:.4f}")
        print(f"  node[{i}] ({_safe_node_name(node_name)}): " + ", ".join(row))


def _print_edges(data, max_edges=20):
    """Print graph edges with edge types."""
    print("Edges:")
    num_edges = data.edge_index.size(1)
    limit = min(max_edges, num_edges)

    for i in range(limit):
        src = data.edge_index[0, i].item()
        dst = data.edge_index[1, i].item()
        edge_type = data.edge_type[i].item() if hasattr(data, "edge_type") else "N/A"
        print(f"  {src} -> {dst} (edge_type={edge_type})")

    if num_edges > limit:
        print(f"  ... {num_edges - limit} more edges")


def example_basic_conversion():
    """Basic usage: convert a tree with engineered features into a graph."""
    print("=" * 60)
    print("Example 1: Basic Tree-to-Graph Conversion")
    print("=" * 60)

    tree = Tree("((A:1,B:2)C:3,D:4)E:0;", format=1)

    engineer = TreeFeatureEngineer(num_time_bins=10)
    tree = engineer.add_features(
        tree,
        origin_time=10.0,
        rescale=False,
        inplace=True,
    )

    converter = TreeToGraphConverter(
        feature_names=engineer.feature_names,
        add_virtual_nodes=False,
        traversal_strategy=engineer.traversal_strategy,
    )

    data = converter.convert(tree)

    _print_data_summary(data, converter)
    print()
    _print_node_features(data, converter)
    print()
    _print_edges(data)
    print()


def example_selective_feature_conversion():
    """Convert a tree using only a selected subset of node features."""
    print("=" * 60)
    print("Example 2: Selective Feature Conversion")
    print("=" * 60)

    tree = Tree("((A:1,B:2)C:3,D:4)E:0;", format=1)

    engineer = TreeFeatureEngineer(num_time_bins=8)
    selected_features = ["node_time", "time_bin", "is_tip", "branch_length"]

    tree = engineer.add_features(
        tree,
        origin_time=10.0,
        feature_names=selected_features,
        rescale=False,
        inplace=True,
    )

    converter = TreeToGraphConverter(
        feature_names=selected_features,
        add_virtual_nodes=False,
        traversal_strategy=engineer.traversal_strategy,
    )

    data = converter.convert(tree)

    print(f"Selected features: {selected_features}")
    _print_data_summary(data, converter)
    print()
    _print_node_features(data, converter)
    print()


def example_virtual_nodes():
    """Convert a tree and add virtual time-bin nodes."""
    print("=" * 60)
    print("Example 3: Conversion with Virtual Time-Bin Nodes")
    print("=" * 60)

    tree = Tree("((A:1,B:1)C:1,D:1)E:0;", format=1)

    engineer = TreeFeatureEngineer(num_time_bins=5)
    tree = engineer.add_features(
        tree,
        origin_time=2.0,
        rescale=False,
        inplace=True,
    )

    converter = TreeToGraphConverter(
        feature_names=engineer.feature_names,
        add_virtual_nodes=True,
        num_time_bins=engineer.num_time_bins,
        traversal_strategy=engineer.traversal_strategy,
        append_is_virtual_feature=True,
    )

    data = converter.convert(tree)

    _print_data_summary(data, converter)
    print()
    _print_node_features(data, converter)
    print()
    _print_edges(data, max_edges=40)

    if hasattr(data, "virtual_node_mask"):
        print("\nVirtual node mask:")
        print(f"  {data.virtual_node_mask.tolist()}")

    if hasattr(data, "node_type"):
        print("\nNode type:")
        print(f"  {data.node_type.tolist()}")

    print()


def example_graph_attributes():
    """Attach graph-level attributes during conversion."""
    print("=" * 60)
    print("Example 4: Graph-Level Attributes")
    print("=" * 60)

    tree = Tree("((A:1,B:2)C:3,D:4)E:0;", format=1)

    engineer = TreeFeatureEngineer(num_time_bins=10)
    tree = engineer.add_features(
        tree,
        origin_time=10.0,
        rescale=True,
        inplace=True,
    )

    converter = TreeToGraphConverter(
        feature_names=engineer.feature_names,
        add_virtual_nodes=False,
        traversal_strategy=engineer.traversal_strategy,
    )

    data = converter.convert(
        tree,
        graph_attrs={
            "tree_id": "example_tree_001",
            "origin_time": 10.0,
            "description": "demo graph with graph-level metadata",
        },
    )

    _print_data_summary(data, converter)
    print("\nGraph attributes:")
    print(f"  tree_id: {data.tree_id}")
    print(f"  origin_time: {data.origin_time}")
    print(f"  description: {data.description}")
    print()


def example_save_and_load():
    """Convert a tree, save the graph, and load it back."""
    print("=" * 60)
    print("Example 5: Save and Load PyG Data")
    print("=" * 60)

    tree = Tree("((A:1,B:1)C:1,D:1)E:0;", format=1)

    engineer = TreeFeatureEngineer(num_time_bins=6)
    tree = engineer.add_features(
        tree,
        origin_time=2.0,
        rescale=False,
        inplace=True,
    )

    converter = TreeToGraphConverter(
        feature_names=engineer.feature_names,
        add_virtual_nodes=True,
        num_time_bins=engineer.num_time_bins,
        traversal_strategy=engineer.traversal_strategy,
    )

    save_path = Path("example_outputs/tree_graph.pt")

    data = converter.convert_and_save(
        tree,
        path=save_path,
        graph_attrs={"tree_id": "saved_tree_example"},
    )

    print(f"Saved graph to: {save_path}")
    _print_data_summary(data, converter)

    loaded_data = TreeToGraphConverter.load_data(save_path)
    print("\nLoaded graph summary:")
    print(f"  num_nodes: {loaded_data.num_nodes}")
    print(f"  num_edges: {loaded_data.edge_index.size(1)}")
    print(f"  x.shape: {tuple(loaded_data.x.shape)}")
    if hasattr(loaded_data, "tree_id"):
        print(f"  tree_id: {loaded_data.tree_id}")
    print()


def main():
    """Run all examples."""
    examples = [
        example_basic_conversion,
        example_selective_feature_conversion,
        example_virtual_nodes,
        example_graph_attributes,
        example_save_and_load,
    ]

    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"Error in {example.__name__}: {e}\n")


if __name__ == "__main__":
    main()
