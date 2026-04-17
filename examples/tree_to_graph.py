"""Self-contained TreeToGraphConverter example."""

from ete3 import Tree

from phylognn.data import TreeFeatureEngineer, TreeToGraphConverter


FEATURE_NAMES = [
    "node_time",
    "time_bin",
    "branch_length",
    "is_tip",
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

    converter = TreeToGraphConverter(
        feature_names=FEATURE_NAMES,
        add_virtual_nodes=False,
    )
    data = converter.convert(tree, graph_attrs={"example_name": "tree_to_graph"})

    print("Graph summary")
    print(f"x shape: {tuple(data.x.shape)}")
    print(f"edge_index shape: {tuple(data.edge_index.shape)}")
    print(f"num_nodes: {data.num_nodes}")
    print(f"num_edges: {data.edge_index.size(1)}")
    print(f"Feature set: {', '.join(FEATURE_NAMES)}")
    print(f"example_name: {data.example_name}")


if __name__ == "__main__":
    main()
