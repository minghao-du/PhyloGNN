def get_max_meta_time(tree):
    """
    获取 ETE3 树中所有带 meta_time 属性节点的最大 meta_time。

    参数:
        tree: ete3 的 Tree 对象

    返回:
        最大 meta_time；如果没有任何节点含有 meta_time，则返回 None
    """

    values = [float(node.meta_time) for node in tree.traverse() if hasattr(node, "meta_time")]
    return max(values) if values else None
