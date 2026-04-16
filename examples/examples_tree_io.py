from phylognn.io import read_tree_as_ete3

ete_tree = read_tree_as_ete3(
    file_path="examples_data/simulated_trees/5.trees",
    schema="nexus",
    tree_index=0,
)

# import dendropy
# from ete3 import Tree


# dtree = dendropy.Tree.get(
#     path="examples_data/simulated_trees/5.trees",
#     schema="nexus",
#     preserve_underscores=True,
#     extract_comment_metadata=True,   # 解析 [&key=value] 这类注释
# )

# for node in dtree.preorder_node_iter():
#     name = node.taxon.label if node.taxon else "internal"
#     length = node.edge_length

#     node_type = node.annotations.get_value("type")
#     samp = node.annotations.get_value("samp")
#     time = node.annotations.get_value("time")

#     print({
#         "name": name,
#         "edge_length": length,
#         "type": node_type,
#         "samp": samp,
#         "time": time,
#     })

# newick = dtree.as_string(
#     schema="newick",
#     suppress_rooting=True,
# )

# etree = Tree(newick, format=1)
