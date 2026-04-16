"""
tree_io.py

工程化的树读取与转换工具：
- 从文件路径读取系统发育树
- 使用 DendroPy 解析
- 转换为 ete3.Tree，便于后续 TreeFeatureEngineer 直接构建特征

设计目标
--------
1. 工程性
   - 提供清晰的 API
   - 完整类型标注
   - 严格参数校验
   - 便于后续扩展和集成

2. 兼容性
   - 支持 nexus / newick 等常见 schema
   - 支持多树文件中按索引读取
   - 支持保留下划线、注释元数据等

3. 可维护性
   - 注释和 specification 清晰
   - 转换逻辑拆分为独立函数
   - 可选保留 DendroPy 注释信息到 ETE3 节点属性

典型使用方式
------------
>>> from tree_io import read_tree_as_ete3
>>> ete_tree = read_tree_as_ete3(
...     file_path="examples_data/simulated_trees/5.trees",
...     schema="nexus",
...     tree_index=0,
... )

然后可直接交给你的 TreeFeatureEngineer：
>>> engineer = TreeFeatureEngineer(num_time_bins=101)
>>> ete_tree = engineer.add_features(ete_tree, origin_time=10.0, rescale=True)

实现说明
--------
1. 多树文件读取
   - 使用 dendropy.TreeList.get(...) 统一读取
   - 通过 tree_index 指定读取第几棵树

2. 节点名称映射
   - 叶节点优先使用 node.taxon.label
   - 若不存在，则退化使用 node.label
   - 内部节点使用 node.label（若存在）

3. 分支长度映射
   - DendroPy: node.edge_length
   - ETE3: node.dist
   - 若 edge_length 为 None，则默认转为 0.0

4. 注释/元数据映射
   - 若 keep_annotations=True，则将 DendroPy annotation 挂到 ETE3 节点
   - 属性名会做安全清洗，避免非法属性名
   - 可选添加统一前缀 annotation_prefix，默认 "meta_"

5. 根节点处理
   - ete3.Tree 本身也是根节点对象
   - 根节点 dist 强制设为 0.0，更符合后续特征工程习惯
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import re

from ete3 import Tree

if TYPE_CHECKING:
    import dendropy


# =========================
# Configuration
# =========================


@dataclass(frozen=True)
class TreeReadConfig:
    """
    树文件读取配置。

    Parameters
    ----------
    schema : str, default="nexus"
        树文件格式，常见值包括：
        - "nexus"
        - "newick"

    tree_index : int, default=0
        若文件中包含多棵树，则读取第几棵树。
        必须满足：
        - tree_index >= 0

    preserve_underscores : bool, default=True
        是否保留下划线，不将其自动解释为空格。

    extract_comment_metadata : bool, default=True
        是否解析类似 [&key=value] 的注释元数据。

    rooting : Optional[str], default=None
        传递给 DendroPy 的 rooting 参数。
        常用值可能包括：
        - None
        - "force-rooted"
        - "force-unrooted"

    keep_annotations : bool, default=True
        是否将 DendroPy 节点注释拷贝到 ETE3 节点属性中。

    annotation_prefix : str, default="meta_"
        注释属性写入到 ETE3 节点时使用的前缀。
        例如注释键为 "rate"，则最终属性名可能为 "meta_rate"。
    """

    schema: str = "nexus"
    tree_index: int = 0
    preserve_underscores: bool = True
    extract_comment_metadata: bool = True
    rooting: Optional[str] = None
    keep_annotations: bool = True
    annotation_prefix: str = "meta_"


# =========================
# Public API
# =========================


def _import_dendropy():
    """Import DendroPy lazily so tree I/O stays optional."""
    try:
        import dendropy
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Tree I/O requires the optional 'dendropy' dependency. "
            'Install it with `pip install -e ".[beast]"` or `pip install dendropy`.'
        ) from exc
    return dendropy


def read_tree_as_ete3(
    file_path: str | Path,
    schema: str = "nexus",
    tree_index: int = 0,
    preserve_underscores: bool = True,
    extract_comment_metadata: bool = True,
    rooting: Optional[str] = None,
    keep_annotations: bool = True,
    annotation_prefix: str = "meta_",
) -> Tree:
    """
    从文件中读取树，并转换为 ete3.Tree。

    Specification
    -------------
    输入：
    - file_path: 树文件路径
    - schema: 文件格式，例如 "nexus" / "newick"
    - tree_index: 当文件包含多棵树时，读取哪一棵
    - preserve_underscores: 是否保留下划线
    - extract_comment_metadata: 是否解析注释元数据
    - rooting: 可选的 rooted / unrooted 解析控制
    - keep_annotations: 是否保留节点注释到 ETE3
    - annotation_prefix: 注释属性前缀

    输出：
    - ete3.Tree 对象

    Guarantees
    ----------
    1. 返回值一定是一个可直接用于遍历和 add_feature 的 ete3.Tree
    2. 节点分支长度被映射到 node.dist
    3. 根节点 dist 被设为 0.0
    4. 叶节点名称优先来自 taxon.label
    5. 若 keep_annotations=True，则 DendroPy 注释会被尽可能保留

    Raises
    ------
    FileNotFoundError
        当文件不存在时抛出。

    ValueError
        当 tree_index 非法、文件中没有树、或 schema 不匹配时抛出。

    RuntimeError
        当底层 DendroPy 解析失败时抛出。
    """
    config = TreeReadConfig(
        schema=schema,
        tree_index=tree_index,
        preserve_underscores=preserve_underscores,
        extract_comment_metadata=extract_comment_metadata,
        rooting=rooting,
        keep_annotations=keep_annotations,
        annotation_prefix=annotation_prefix,
    )

    dtree = read_tree_with_dendropy(file_path=file_path, config=config)
    return dendropy_tree_to_ete3(dtree=dtree, config=config)


def read_tree_with_dendropy(
    file_path: str | Path,
    config: TreeReadConfig,
) -> "dendropy.Tree":
    """
    使用 DendroPy 从文件中读取单棵树。

    Implementation contract
    -----------------------
    - 使用 TreeList.get(...) 统一处理单树文件/多树文件
    - 通过 config.tree_index 选择目标树
    - 若文件解析失败，抛出工程化异常信息
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Tree file does not exist: {path}")

    if not path.is_file():
        raise FileNotFoundError(f"Path is not a file: {path}")

    if config.tree_index < 0:
        raise ValueError(f"tree_index must be >= 0, got {config.tree_index}")

    dendropy = _import_dendropy()

    try:
        tree_list = dendropy.TreeList.get(
            path=str(path),
            schema=config.schema,
            preserve_underscores=config.preserve_underscores,
            extract_comment_metadata=config.extract_comment_metadata,
            rooting=config.rooting,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to parse tree file '{path}' with schema='{config.schema}'."
        ) from exc

    if len(tree_list) == 0:
        raise ValueError(f"No trees found in file: {path}")

    if config.tree_index >= len(tree_list):
        raise ValueError(
            f"tree_index out of range: {config.tree_index}. "
            f"File contains {len(tree_list)} tree(s)."
        )

    return tree_list[config.tree_index]


def dendropy_tree_to_ete3(
    dtree: "dendropy.Tree",
    config: Optional[TreeReadConfig] = None,
) -> Tree:
    """
    将 DendroPy Tree 转换为 ete3.Tree。

    Parameters
    ----------
    dtree : dendropy.Tree
        已经解析好的 DendroPy 树对象。

    config : Optional[TreeReadConfig], default=None
        转换配置。
        若为 None，则使用默认配置。

    Returns
    -------
    ete3.Tree
        转换后的 ETE3 树对象。

    Conversion rules
    ----------------
    1. 结构
       - 保持父子拓扑结构一致

    2. 名称
       - 叶节点：优先 taxon.label，其次 node.label
       - 内部节点：使用 node.label（若存在）

    3. 分支长度
       - node.edge_length -> ete_node.dist
       - None -> 0.0

    4. 注释元数据
       - 若 keep_annotations=True，则复制为 ETE3 节点属性
       - 属性名通过 _sanitize_feature_name 清洗

    5. 根节点
       - 返回的根节点类型为 ete3.Tree
       - 根节点 dist 固定为 0.0
    """
    if config is None:
        config = TreeReadConfig()

    if dtree.seed_node is None:
        raise ValueError("Input DendroPy tree has no seed_node/root.")

    ete_tree = Tree()
    _populate_ete_node(
        dnode=dtree.seed_node,
        ete_node=ete_tree,
        config=config,
        is_root=True,
    )
    return ete_tree


# =========================
# Internal helpers
# =========================


def _populate_ete_node(
    dnode: "dendropy.Node",
    ete_node: Tree,
    config: TreeReadConfig,
    is_root: bool = False,
) -> None:
    """
    递归填充单个 ETE3 节点及其全部子树。

    Parameters
    ----------
    dnode : dendropy.Node
        当前 DendroPy 节点。

    ete_node : ete3.Tree
        当前对应的 ETE3 节点对象。

    config : TreeReadConfig
        转换配置。

    is_root : bool, default=False
        当前节点是否为根节点。
    """
    ete_node.name = _extract_node_name(dnode)
    ete_node.dist = 0.0 if is_root else _safe_edge_length(dnode.edge_length)

    # 可选保留注释到节点属性，便于后续调试、分析或特征扩展
    if config.keep_annotations:
        _copy_dendropy_annotations_to_ete_node(
            dnode=dnode,
            ete_node=ete_node,
            annotation_prefix=config.annotation_prefix,
        )

    # 附加一些常用原始信息，增强工程可追踪性
    _attach_basic_raw_metadata(dnode=dnode, ete_node=ete_node)

    for child_dnode in dnode.child_node_iter():
        child_ete = ete_node.add_child()
        _populate_ete_node(
            dnode=child_dnode,
            ete_node=child_ete,
            config=config,
            is_root=False,
        )


def _extract_node_name(dnode: "dendropy.Node") -> str:
    """
    提取节点名称。

    命名策略
    --------
    1. 若为叶节点且存在 taxon.label，则优先使用 taxon.label
    2. 否则若 node.label 存在，则使用 node.label
    3. 否则返回空字符串
    """
    if dnode.taxon is not None and dnode.taxon.label is not None:
        return str(dnode.taxon.label)

    if dnode.label is not None:
        return str(dnode.label)

    return ""


def _safe_edge_length(length: Optional[float]) -> float:
    """
    将 DendroPy 的 edge_length 转换为安全的 float。

    Rules
    -----
    - None -> 0.0
    - 其他数值 -> float(length)
    """
    if length is None:
        return 0.0
    return float(length)


def _copy_dendropy_annotations_to_ete_node(
    dnode: "dendropy.Node",
    ete_node: Tree,
    annotation_prefix: str = "meta_",
) -> None:
    """
    将 DendroPy 节点 annotations 拷贝到 ETE3 节点属性。

    Notes
    -----
    1. 属性名会经过清洗，保证适合作为 Python attribute
    2. 若注释值无法直接序列化，则退化为字符串
    3. 若属性名冲突，后写入的值会覆盖前值
    """
    try:
        annotations = getattr(dnode, "annotations", None)
        if annotations is None:
            return

        for ann in annotations:
            raw_key = getattr(ann, "name", None)
            raw_value = getattr(ann, "value", None)

            if raw_key is None:
                continue

            feature_name = _sanitize_feature_name(f"{annotation_prefix}{raw_key}")
            feature_value = _normalize_annotation_value(raw_value)

            ete_node.add_feature(feature_name, feature_value)

    except Exception:
        # 保持转换过程稳健：注释复制失败不影响主流程
        return


def _attach_basic_raw_metadata(dnode: "dendropy.Node", ete_node: Tree) -> None:
    """
    附加少量基础原始元信息，提升工程可追踪性。

    当前附加字段
    ----------
    raw_node_label : Optional[str]
        DendroPy 原始 node.label

    raw_taxon_label : Optional[str]
        DendroPy 原始 taxon.label

    has_taxon : int
        是否存在 taxon
    """
    raw_node_label = dnode.label if dnode.label is not None else None
    raw_taxon_label = (
        dnode.taxon.label if (dnode.taxon is not None and dnode.taxon.label is not None) else None
    )

    ete_node.add_feature("raw_node_label", raw_node_label)
    ete_node.add_feature("raw_taxon_label", raw_taxon_label)
    ete_node.add_feature("has_taxon", 1 if dnode.taxon is not None else 0)


def _sanitize_feature_name(name: str) -> str:
    """
    将任意字符串清洗为适合作为 ETE3 节点属性名的形式。

    规则
    ----
    - 非字母数字下划线字符替换为下划线
    - 若首字符不是字母或下划线，则前置一个下划线
    - 连续下划线压缩为单个下划线
    """
    sanitized = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    sanitized = re.sub(r"_+", "_", sanitized)

    if not sanitized:
        return "_"

    if not re.match(r"[a-zA-Z_]", sanitized[0]):
        sanitized = f"_{sanitized}"

    return sanitized


def _normalize_annotation_value(value: Any) -> Any:
    """
    规范化 annotation value，尽量返回稳定、可读、可挂载的对象。

    策略
    ----
    - 基本标量类型直接返回
    - 其他对象退化为字符串
    """
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    return str(value)
