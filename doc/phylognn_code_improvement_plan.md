# PhyloGNN 代码补充与整改方案

## 目标

本文档基于当前 `src/phylognn/` 的代码结构和已有测试情况，整理出需要补充和修改的部分，并给出一套可执行的整改方案。重点不是做大范围重构，而是先把主流程契约、测试保护和模块一致性补齐。

## 当前总体判断

当前仓库已经具备以下基础：

- `data` 层已经形成相对清晰的两阶段流水线：
  `TreeFeatureEngineer -> TreeToGraphConverter`
- `training` 层已经开始工程化，具备 dataset / trainer / metrics 的基本结构
- `models` 层已经有较完整的基类与 `GATBiLSTMNet`
- `tree_io` 提供了从文件读取系统发育树的独立入口

但目前模块成熟度不一致，主要问题集中在：

- 部分核心行为的实现与文档不一致
- 数据层和模型层之间的契约没有完全闭环
- `MultiTaskGATNet` 仍保留旧式假设
- 测试文件明显落后于当前 API
- 包导出层与文档层存在漂移

## 主要问题清单

### 1. `TreeFeatureEngineer` 的 `rescale` 契约不完整

当前 `add_features(..., rescale=True)` 的文档表述是：

- 分支长度会缩放
- `origin_time` 也会按相同比例缩放
- 后续特征计算基于缩放后的值

但当前实现实际上只缩放了 `node.dist`，没有同步缩放参与后续计算的 `origin_time`。

这会直接影响以下特征的正确性：

- `node_time`
- `time_bin`
- `is_fossil`
- `is_extant`
- 以及依赖这些字段的其他派生特征

这属于核心行为 bug，应优先修复。

### 2. `TreeToGraphConverter` 与模型输入契约未闭环

当前 `TreeToGraphConverter` 会把 `time_bin` 放进 `data.x`，但不会显式写入 `data.time_bin`。

与此同时，`GATBiLSTMNet` 在 `temporal_mode != "none"` 时明确要求：

- `data.time_bin` 存在
- `data.time_bin` 为 `LongTensor`
- 长度与节点数匹配

这导致当前标准流水线：

1. 树加特征
2. 图转换
3. 模型前向

并不能天然打通，使用者还需要自己从 `x` 中再拆出 `time_bin`。

这说明数据层输出协议还需要补齐。

### 3. `MultiTaskGATNet` 仍依赖旧式特征列位置假设

当前 `MultiTaskGATNet` 的实现存在几个问题：

- 假定 `x[:, 1]` 就是 time bin
- 对 `feature_names` 顺序强耦合
- 与当前 `TreeToGraphConverter` 的可配置特征顺序设计不兼容
- 类型导入不完整，直接使用了 `Dict` 但未导入
- 缺少与 `BaseGATNet` / `GATBiLSTMNet` 一致的输入校验和契约说明

这个模块目前更像早期研究代码，而不是和当前仓库其他工程化模块同等级的实现。

### 4. 测试集已经明显过时

当前 `tests/` 下的内容不能有效保护现在的代码：

- `tests/test_data_conversion.py` 还在断言节点上存在旧式 `features` 列表
- 该测试逻辑与当前 attribute-based feature API 不一致
- `tests/test_feature_engineer.py` 中关于 `rescale_tree` 的预期签名与现实现状不一致

这意味着：

- 测试无法真实反映当前行为
- 即使代码继续修改，也缺少可靠的回归保护
- 文档和代码漂移后，测试并没有起到纠偏作用

### 5. 导出层和包接口仍有明显缺口

目前包导出层至少存在以下问题：

- `src/phylognn/training/__init__.py` 使用了 `all = [...]`，而不是 `__all__`
- 其中还导出了并不存在的名字 `PhyloDataset`
- 根包 `src/phylognn/__init__.py` 只导出了 `TreeFeatureEngineer` 和 `TreeToGraphConverter`
- 各子模块对外暴露的 API 没有完全统一

这会影响：

- `from phylognn.training import *` 等导入行为
- 文档中对公开 API 的表达
- 使用者对哪些对象是稳定接口的判断

### 6. 文档和示例存在漂移

当前文档和示例中有部分内容已经和实现不一致，例如：

- `src/phylognn/data/__init__.py` 中 custom feature 示例仍沿用旧签名
- `custom_features` 的示例写法和真实实现要求不一致
- 一些说明没有准确反映 `TreeFeatureEngineer` / `TreeToGraphConverter` 当前契约

如果这些文档不更新，会持续误导后续开发和使用。

## 建议修改方案

## 第一阶段：修主链契约

这一阶段只处理最关键、最影响实际可用性的部分。

### 1. 修复 `TreeFeatureEngineer.add_features()` 的 rescale 语义

建议做法：

- 在 `rescale=True` 时，明确把 `origin_time` 按同样比例缩放
- 后续所有依赖时间的特征都统一使用缩放后的 `origin_time`
- 保持 `rescale_factor` 的行为不变

需要补充的测试：

- `rescale=True` 时 `node_time` 是否基于缩放后 `origin_time`
- `time_bin` 是否与缩放后一致
- `is_extant` / `is_fossil` 边界节点是否判定正确
- `rescale=False` 时行为是否保持原样

验收标准：

- 文档定义与实际实现一致
- 相关测试覆盖 rescale 开关两种路径

### 2. 在 `TreeToGraphConverter` 中显式生成 `data.time_bin`

建议做法：

- 如果 `feature_names` 中包含 `time_bin`，在 `convert()` 结束前显式写入 `data.time_bin`
- 确保其 dtype 为 `torch.long`
- 若使用 virtual nodes，需要明确 virtual node 的 `time_bin` 规则
- 该字段应与 `data.x` 中的 `time_bin` 列保持一致

可选增强：

- 同时写入 `data.feature_names`
- 或写入 `data.feature_name_to_index`

这样模型层就不需要再猜某个字段在 `x` 的哪一列。

验收标准：

- `TreeToGraphConverter.convert()` 输出的 `Data` 能直接被 `GATBiLSTMNet` 的 temporal 模式消费
- 不再需要用户手工从 `x` 中提取 `time_bin`

## 第二阶段：统一模型层契约

### 3. 重构 `MultiTaskGATNet`

建议目标：

- 不再依赖 `x[:, 1]` 这种硬编码列索引
- 改为统一读取 `data.time_bin`
- 增加和 `GATBiLSTMNet` 一致的输入校验
- 修复类型导入、构造参数校验、模块重置逻辑
- 明确多任务输出字典与 trainer 约定之间的关系

建议改造内容：

- 增加 `num_time_bins` 参数
- 增加 `_validate_temporal_data()` 或复用已有校验风格
- 增加 `get_encoder_modules()` 和 `get_head_modules()`
- 增加 `reset_parameters()`
- 为 `task_configs` 补强验证逻辑

验收标准：

- `MultiTaskGATNet` 与 `Trainer` 的多任务接口约定一致
- `MultiTaskGATNet` 不再依赖特征列顺序
- 可以使用 `TreeToGraphConverter` 的标准输出直接前向

## 第三阶段：重建测试保护

### 4. 重写并补充测试

建议按模块拆分测试，而不是继续延续当前混合、过时的结构。

建议测试布局：

- `tests/test_feature_engineer.py`
- `tests/test_converter.py`
- `tests/test_tree_io.py`
- `tests/test_models_gat_lstm.py`
- `tests/test_models_multitask.py`
- `tests/test_training_dataset.py`
- `tests/test_training_trainer.py`
- `tests/test_end_to_end_pipeline.py`

重点补充内容如下。

#### `TreeFeatureEngineer`

- 参数校验
- custom feature 注册
- feature dependency 自动补齐
- `rescale=True` / `False`
- `time_bin` 边界
- `is_fossil` / `is_extant`
- `inplace=True` / `False`

#### `TreeToGraphConverter`

- 基本图转换
- `edge_index` 方向性
- `edge_type`
- `node_names`
- `virtual_node_mask`
- virtual nodes 的 `time_bin`
- `data.time_bin` 输出
- 保存与加载

#### `tree_io`

- 单树读取
- 多树文件按索引读取
- schema 错误
- 文件不存在
- annotation 保留
- root / edge length 映射

#### 模型测试

- `GATBiLSTMNet` 的 `none / fc / lstm` 三种 temporal mode
- 缺失 `data.time_bin` 时抛错
- 多任务模型输出 shape
- batched graph 前向

#### 端到端测试

- `Tree -> add_features -> convert -> model.forward`
- 单任务最小训练循环
- 多任务最小训练循环

验收标准：

- 测试内容围绕当前 API，而不是历史接口
- 每个核心模块都至少有一组正常路径和一组错误路径测试

## 第四阶段：修包接口与文档层

### 5. 修正导出层

建议修改：

- 把 `src/phylognn/training/__init__.py` 中的 `all` 改为 `__all__`
- 删除不存在的导出名
- 检查 `data` / `models` / `training` / 根包 `__init__.py` 的对外接口是否一致
- 明确哪些对象属于稳定公共 API

验收标准：

- 所有 `__all__` 与真实可导入对象一致
- 不再暴露不存在或过时的名字

### 6. 同步 docstring 和模块说明

建议修改：

- 更新 `src/phylognn/data/__init__.py` 示例
- 更新 custom feature 的示例签名
- 校对 `TreeFeatureEngineer` 和 `TreeToGraphConverter` 的 docstring
- 保证示例代码与真实 API 一致

验收标准：

- 文档不再依赖旧接口
- docstring 能直接作为最小使用说明

## 第五阶段：补一个标准端到端示例

### 7. 增加推荐使用范式

建议在 `examples/` 中保留一个标准示例，串起完整流程：

1. 从文件读取树
2. 调用 `TreeFeatureEngineer`
3. 调用 `TreeToGraphConverter`
4. 构造 dataset / dataloader
5. 构造模型
6. 运行 `Trainer.fit()`

建议示例文件：

- `examples/minimal_pipeline.py`
- 或 `examples/end_to_end_training.py`

验收标准：

- 使用者可以通过一个示例看到仓库推荐工作流
- 示例不依赖过时接口

## 推荐实施顺序

建议按以下顺序推进：

1. 修复 `TreeFeatureEngineer` 的 rescale 契约
2. 在 `TreeToGraphConverter` 中补齐 `data.time_bin`
3. 重构 `MultiTaskGATNet`
4. 重建测试
5. 修正导出层
6. 更新 docstring 与示例
7. 增加端到端 example

这个顺序的原因是：

- 先修核心数学与数据契约
- 再修模型消费端
- 最后用测试和文档把新契约稳定下来

## 具体实施任务拆分

如果按开发任务来拆，建议分成以下几个 PR 或阶段提交。

### 任务 A：特征工程契约修复

- 修改 `TreeFeatureEngineer.add_features()`
- 补 rescale 相关测试
- 校正文档表述

### 任务 B：图数据协议补齐

- 修改 `TreeToGraphConverter.convert()`
- 增加 `data.time_bin`
- 补 converter 相关测试

### 任务 C：多任务模型对齐

- 重构 `MultiTaskGATNet`
- 增加多任务模型测试
- 确保和 `Trainer` 协议一致

### 任务 D：测试体系清理

- 删除或重写旧接口测试
- 新增模块化测试文件
- 增加端到端 smoke test

### 任务 E：公共接口与文档整理

- 修 `__all__`
- 清理错误示例
- 更新模块 docstring
- 增加正式 example

## 验收清单

整改完成后，建议至少满足以下验收条件：

- `TreeFeatureEngineer` 在 `rescale=True` 下行为与文档一致
- `TreeToGraphConverter` 输出可直接供 temporal model 使用
- `MultiTaskGATNet` 不依赖硬编码特征列顺序
- `tests/` 中不再保留基于旧接口的主测试逻辑
- `training/__init__.py` 等导出层正确使用 `__all__`
- 文档示例可直接反映当前推荐用法
- 存在一个最小端到端示例作为参考实现

## 建议优先级

### P0

- 修复 `rescale` 语义
- 补 `data.time_bin`
- 让主链可以直接跑通

### P1

- 重构 `MultiTaskGATNet`
- 更新测试到当前 API

### P2

- 修导出层
- 修文档与示例
- 补标准 end-to-end example

## 总结

当前仓库不是“缺少很多新模块”，而是“已有模块之间的契约还没有完全统一”。因此最应该补的不是更多功能，而是以下三类基础工作：

- 把核心行为定义和实现对齐
- 把数据层输出和模型层输入对齐
- 用测试和文档把这些契约固定下来

只要先把这三件事补齐，后续无论继续扩展特征工程、加模型、还是做训练工作流，代码都会稳定很多。
