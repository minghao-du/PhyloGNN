from pathlib import Path
import random
import torch
from torch_geometric.loader import DataLoader

from phylognn.data import (
    TreeFeatureEngineer,
    TreeToGraphConverter,
)
from phylognn.io import read_tree_as_ete3
from phylognn.utils import get_max_meta_time
from phylognn.training import SplitPhyloDiskDataset, DatasetSplit


# ============================================================
# 基础配置
# ============================================================

# 输入目录：存放所有树文件
INPUT_DIR = Path("examples_data/simulated_trees")

# 输出目录
OUTPUT_X_DIR = Path("example_outputs/x")
OUTPUT_Y_DIR = Path("example_outputs/y")
SPLIT_DIR = Path("example_outputs/splits")

# 读取树时使用的 schema
TREE_SCHEMA = "nexus"

# 如果一个文件中包含多个 tree，这里默认读取第 0 棵
TREE_INDEX = 0

# 时间分箱数量
NUM_TIME_BINS = 10

# 是否固定随机种子（方便复现）
RANDOM_SEED = 42


# ============================================================
# 工具函数
# ============================================================


def ensure_directories() -> None:
    """
    创建输出目录（如果不存在则自动创建）。
    """
    OUTPUT_X_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_Y_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)


def list_tree_files(input_dir: Path) -> list[Path]:
    """
    自动收集输入目录下的所有树文件。

    说明：
    - 这里默认读取该目录下的所有“文件”，不递归子目录。
    - 如果你后续只想限制某些后缀（如 .trees / .tree / .nexus），
      可以在这里自行加过滤逻辑。
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    tree_files = sorted([p for p in input_dir.iterdir() if p.is_file()])

    if not tree_files:
        raise FileNotFoundError(f"在目录中未找到任何树文件: {input_dir}")

    return tree_files


def build_processors():
    """
    构建特征工程器和图转换器。

    说明：
    - 先创建 feature_engineer，并获取全部可用特征名称；
    - add_features 时沿用你原来的逻辑：available_features[1:]
    - convert 时沿用你原来的逻辑：feature_names=available_features
    """
    feature_engineer = TreeFeatureEngineer(num_time_bins=NUM_TIME_BINS)
    available_features = feature_engineer.get_available_features()

    converter = TreeToGraphConverter(
        feature_names=available_features,
        add_virtual_nodes=True,
    )

    return feature_engineer, available_features, converter


def process_single_tree_file(
    tree_file: Path,
    feature_engineer,
    available_features,
    converter,
) -> None:
    """
    处理单个树文件，并保存对应的 x 和 y。

    输出规则：
    - 图数据保存为：example_outputs/x/<文件名去后缀>.pt
    - 标签保存为：example_outputs/y/<文件名去后缀>.pt
    """
    # --------------------------------------------------------
    # 1) 读取树
    # --------------------------------------------------------
    ete_tree = read_tree_as_ete3(
        file_path=str(tree_file),
        schema=TREE_SCHEMA,
        tree_index=TREE_INDEX,
    )

    # 获取树的 origin_time
    origin_time = get_max_meta_time(ete_tree)

    # --------------------------------------------------------
    # 2) 特征工程
    # --------------------------------------------------------
    tree_with_features = feature_engineer.add_features(
        ete_tree,
        origin_time=origin_time,
        feature_names=available_features[1:],  # 按你原始逻辑：添加除第一个以外的全部特征
        rescale=True,
        inplace=True,
    )

    # --------------------------------------------------------
    # 3) 转图并保存 x
    # --------------------------------------------------------
    file_stem = tree_file.stem
    x_save_path = OUTPUT_X_DIR / f"{file_stem}.pt"

    converter.convert_and_save(
        tree_with_features,
        path=x_save_path,
    )

    # --------------------------------------------------------
    # 4) 随机生成 y 标签并保存
    # --------------------------------------------------------
    ylabel = random.randint(0, 1)
    ylabel_tensor = torch.tensor(ylabel, dtype=torch.long)

    y_save_path = OUTPUT_Y_DIR / f"{file_stem}.pt"
    torch.save(ylabel_tensor, y_save_path)

    print(f"[SUCCESS] 已处理: {tree_file.name} -> x: {x_save_path.name}, y: {y_save_path.name}")


def build_dataset_and_splits():
    graph_dir = OUTPUT_X_DIR
    label_dir = OUTPUT_Y_DIR
    split_dir = SPLIT_DIR

    # 1) 先构建完整数据集
    dataset = SplitPhyloDiskDataset(
        graph_dir=graph_dir,
        label_dir=label_dir,
        recursive=False,  # 如果 x/y 下面没有子目录，就保持 False
        load_on_cpu=True,
        cache_graphs=False,
        cache_labels=False,
        strict_label_check=True,  # 图和标签必须一一对应，建议开着
        sample_id_from_relative_path=True,
    )

    # 2) 如果已经有划分文件，则直接读取，保证以后始终一致
    if split_dir.exists():
        split = DatasetSplit.from_manifest_dir(split_dir)

    # 3) 否则第一次生成，并立刻保存到磁盘
    else:
        split = DatasetSplit.from_ratios(
            sample_ids=dataset.sample_ids,
            train_ratio=7 / 10,
            val_ratio=2 / 10,
            test_ratio=1 / 10,
            seed=42,  # 固定随机种子，保证可复现
            shuffle=True,
        )

        dataset.export_split_manifests(split, split_dir)

    # 4) 构建 train / val / test 子集
    subsets = dataset.build_subsets(split)

    train_dataset = subsets["train"]
    val_dataset = subsets["val"]
    test_dataset = subsets["test"]

    return dataset, split, train_dataset, val_dataset, test_dataset


def main() -> None:
    """
    主流程：
    1. 初始化环境
    2. 收集所有树文件
    3. 逐个处理并保存
    """
    # 固定随机种子，保证随机标签可复现
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # # 创建输出目录
    # ensure_directories()

    # # 获取所有树文件
    # tree_files = list_tree_files(INPUT_DIR)

    # # 初始化处理器
    # feature_engineer, available_features, converter = build_processors()

    # print(f"共检测到 {len(tree_files)} 个树文件，开始处理...")

    # success_count = 0
    # failed_files = []

    # for tree_file in tree_files:
    #     try:
    #         process_single_tree_file(
    #             tree_file=tree_file,
    #             feature_engineer=feature_engineer,
    #             available_features=available_features,
    #             converter=converter,
    #         )
    #         success_count += 1
    #     except Exception as e:
    #         failed_files.append((tree_file.name, str(e)))
    #         print(f"[FAILED] 处理失败: {tree_file.name} | 原因: {e}")

    # # 输出汇总结果
    # print("\n================ 处理完成 ================")
    # print(f"成功: {success_count}")
    # print(f"失败: {len(failed_files)}")

    # if failed_files:
    #     print("失败文件列表：")
    #     for file_name, err in failed_files:
    #         print(f" - {file_name}: {err}")

    dataset, split, train_dataset, val_dataset, test_dataset = build_dataset_and_splits()

    print(dataset)
    print(split.split_names())
    print(f"total: {len(dataset)}")
    print(f"train: {len(train_dataset)}")
    print(f"val:   {len(val_dataset)}")
    print(f"test:  {len(test_dataset)}")

    _train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    _val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    _test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)


if __name__ == "__main__":
    main()
