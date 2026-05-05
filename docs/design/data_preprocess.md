# AdaManip Data Preprocess: Language Embedding Index

本文档定义数据预处理模块（独立于采集）的语言向量生成与索引规范。

目标：不改采集主流程与 zarr 主结构，在训练前新增一个预处理步骤，基于已有语言标注构建 embedding 字典，训练时通过 key 快速索引。

## 1. 模块边界

1. 数据采集模块保持不变。
2. 预处理模块读取采集输出（如 language_expanded.json、trajectory_language.jsonl、frame_language.jsonl）。
3. 预处理模块输出一个 embedding 字典文件，供训练阶段加载。

## 2. 输入来源

预处理模块读取：

1. expanded_minimal_chains（用于 chain 级索引）
2. operation_set（用于 operation 级索引）

文本来源约定：

1. chain 文本：将一条 chain 的步骤按固定分隔符拼接为一句文本。
2. operation 文本：直接使用 operation 字符串。

编码模型可选：

1. sentence-transformers/all-MiniLM-L6-v2（384 维）
2. m3e-small（512 维）

## 3. 唯一输出格式（单字典）

预处理模块只需要保存一个字典，格式如下：

```json
{
  "encoder": {
    "name": "moka-ai/m3e-small",
    "output_dim": 512,
    "normalized": true
  },
  "expanded_minimal_chains": [embedding1, embedding2, ...],
  "operation_set": {
    "operation1": embedding1,
    "operation2": embedding2
  }
}
```

说明：

1. expanded_minimal_chains 的下标就是 chain id。
2. operation_set 的 key 是 operation 字符串。
3. embedding 类型可为 list[float]（json）或 ndarray（若二进制序列化）。

## 4. 推荐落盘文件

建议文件名：

1. language_embedding_dict.json（json 可读）

若 embedding 维度较大、json 体积过大，可使用等价二进制格式（如 npy/pth），但逻辑结构必须与第 3 节一致。

## 5. 训练时索引方式

训练流程不需要文本编码器在线参与，只需按 key 索引：

1. chain 条件：
   - 从原始数据得到 chain id
   - embedding = dict["expanded_minimal_chains"][chain_id]

2. operation 条件：
   - 从原始数据得到 operation 字符串
   - embedding = dict["operation_set"][operation_str]

这样可避免 batch 内重复文本编码，降低训练开销。

## 6. 一致性约束

1. expanded_minimal_chains 的长度必须等于 language_expanded.json 中 chain 数量。
2. operation_set 的 key 集合必须与 language_expanded.json 中 operation_set 一致。
3. 同一个预处理产物内，chain embedding 与 operation embedding 必须来自同一个文本编码器与同一版本权重。
4. 预处理脚本应记录 encoder_name 与 embedding_dim（可作为文件附加元信息，或并行写入 metadata 文件）。

## 7. 最小验收

1. 预处理阶段产出单字典文件。
2. 训练脚本可仅通过 chain id 或 operation 字符串完成 embedding 索引。
3. 不依赖在线文本编码即可完成训练数据读取。
4. 与采集文档解耦：采集侧无需新增语言 embedding 存储字段。
