# AdaManip Data Collection Language Schema (Final)

本文档定义数据采集阶段的语言标注存储规范。

本版本采用单一确定方案：外挂文件，不改 zarr 主结构。

## 1. 设计目标

1. 提供统一任务语言模板配置，结构化记录任务命令、操作集合、抽象步骤链。
2. 采集时按任务名读取模板，并在输出数据目录写入本次采集对应的语言文件。
3. 含 `Nx` 的链在采集后按实际轨迹统计进行离散展开，避免训练时动态解析。
4. 同时保存三层语言标注：任务级、轨迹级、帧级。

## 2. 输出文件（外挂）

每个采集数据目录建议包含：

1. `demo_data.zip`
- 现有 zarr 数据，保持不变：`data/pcs`, `data/env_state`, `data/action`, `meta/episode_ends`。

2. `language_expanded.json`
- 本次采集后生成的展开配置（根据实际轨迹得出离散链）。
- 必须落盘到当前数据目录，保证可复现。

3. `trajectory_language.jsonl`
- 文件内容是一个列表（List），每个元素是一个轨迹记录字典。

4. `frame_language.jsonl`
- 文件内容是一个列表（List），每个元素是一个帧记录字典。

统一模板文件位置（非输出目录）：
- `third_party/ada_manip/cfg/language_template.json`
- 该文件作为所有任务共享配置，采集时从该路径读取。

## 3. 为什么模板不包含 expand 属性

结论：模板层不放 `expand`、`n_min/n_max`、`expanded`。

原因：
1. 对部分任务，`N` 的有效范围依赖真实采集执行，不应在采集前硬编码。
2. 若模板预填错误范围，会污染数据并引入二次修订成本。
3. 模板用于“输入定义”，展开用于“输出事实”，两者应解耦。

因此采用两阶段：
1. 采集前读取模板（抽象链，允许 `Nx` 占位）。
2. 采集后根据实际轨迹统计生成 `language_expanded.json`（事实链，无 `Nx`）。

## 4. 统一任务语言模板格式

统一使用 JSON。模板文件固定路径为 `third_party/ada_manip/cfg/language_template.json`。示例：

```json
{
  "schema_version": "v1",
  "tasks": {
    "bottle": {
      "command": "打开瓶子",
      "operation_set": ["旋转瓶盖", "向上提起瓶盖"],
      "minimal_chains": [
        "向上提起瓶盖",
        "Nx旋转瓶盖 -> 向上提起瓶盖"
      ]
    }
  }
}
```

## 5. 采集后展开配置格式

`language_expanded.json` 记录本次任务采集实际出现的全部最优（无冗余步骤）离散链（全部轨迹的minimal_chain的集合）：

```json
{
  "schema_version": "v1",
  "generated_from": "language_template.json",
  "tasks": {
    "bottle": {
      "command": "打开瓶子",
      "operation_set": ["旋转瓶盖", "向上提起瓶盖"],
      "expanded_minimal_chains": [
        ["向上提起瓶盖"],
        ["1x旋转瓶盖", "向上提起瓶盖"],
        ["2x旋转瓶盖", "向上提起瓶盖"],
        ["3x旋转瓶盖", "向上提起瓶盖"]
      ]
    }
  }
}
```

约定：`expanded_minimal_chains` 的索引即最短链 id（从 0 开始）。

## 6. 轨迹级语言标注（双链）

每条轨迹必须保存两类链：

1. `minimal_chain`
- 最短归约链，对应 `minimal_chain_id` 与链内容（用于类别/任务级监督）。

2. `minimal_chain_id`
- int，表示 `minimal_chain` 在任务级 `expanded_minimal_chains` 中的索引（从 0 开始）。

3. `attempt_chain`
- 完整尝试记录，按真实执行顺序逐条写出（可包含失败后重复尝试）。
- `attempt_chain` 不设置 id。

4. `stage_status`
- 与 `attempt_chain` 等长的 bool 列表。
- `True` 表示该 stage 成功，`False` 表示该 stage 失败。
- 对成功轨迹，最后一个 `stage_status` 必须为 `True`。

5. `command_chains`
- 供训练时直接随机采样的候选命令链集合。
- 类型为 `List[List[str]]`。

6. `command_chain_ids`
- `command_chains` 的索引映射，类型为 `List[int]`。
- 每个元素表示对应命令链在任务级 `expanded_minimal_chains` 中的索引。
- 生成逻辑固定如下：
  1. 从 `attempt_chain` 中找到第一个失败位置 `k`；若无失败，则 `k` 为末尾。
  2. 取子链 `prefix = attempt_chain[:k+1]`（若无失败则为整条链）。
  3. 在任务级 `expanded_minimal_chains` 中，筛选所有以 `prefix` 为起始的链，组成 `command_chains`。
  4. 同步记录这些链在 `expanded_minimal_chains` 中的索引为 `command_chain_ids`。
  5. `command_chains` 必须非空；若为空，说明任务逻辑或标注存在错误，必须修复。

说明：
- 该规则体现“命令一致性优先”：当某 stage 被判成功时，后续动作应继续遵守任务命令；首个失败之后允许轨迹出现与命令不一致的尝试分支，因此命令链只锚定到“首个失败及其之前”的前缀。

示例：

```json
{
  "episode_id": 17,
  "task": "bottle",
  "command": "打开瓶子",
  "minimal_chain_id": 1,
  "minimal_chain": ["1x旋转瓶盖", "向上提起瓶盖"],
  "attempt_chain": ["1x旋转瓶盖", "向上提起瓶盖", "1x旋转瓶盖", "向上提起瓶盖"],
  "stage_status": [true, false, true, true],
  "command_chains": [
    ["1x旋转瓶盖", "向上提起瓶盖"]
  ],
  "command_chain_ids": [1],
  "frame_range": [1042, 1099],
  "success": true
}
```

确定化约定：`frame_range` 使用半开区间 `[start, end)`。

## 7. 帧级语言标注（operation-only）

帧标签使用 `step_operation`，直接等于原子操作名，不带 `Nx`。

示例：

```json
{
  "step_index": 0,
  "step_operation": "旋转瓶盖"
}
```

字段约定：
1. `step_index` 指向 `attempt_chain` 的当前步骤索引（从 0 开始）。
2. `step_operation` 为原子操作名（属于 `operation_set`）。

## 8. Bottle 任务示例（端到端）

### 8.1 抽象模板
- `minimal_chains`:
  - `向上提起瓶盖`
  - `Nx旋转瓶盖 -> 向上提起瓶盖`

### 8.2 采集后展开
- 实际出现 `N = 1, 2, 3`，写入 `language_expanded.json`。

### 8.3 轨迹记录
- 某条轨迹归约到 `expanded_minimal_chains[2]`。
- `minimal_chain_id = 2`。
- `minimal_chain = ["2x旋转瓶盖", "向上提起瓶盖"]`。
- 若发生重复尝试，`attempt_chain` 记录完整序列，`stage_status` 同步记录每个 stage 成败。
- `command_chains` 由“首个失败及其之前前缀”在任务级 `expanded_minimal_chains` 中做前缀匹配得到，`command_chain_ids` 记录对应索引。

### 8.4 帧记录
- 所有旋转阶段帧：`step_operation = "旋转瓶盖"`。
- 所有提起阶段帧：`step_operation = "向上提起瓶盖"`。

## 9. 最小验收标准

1. 数据目录中存在 `language_expanded.json`。
2. `trajectory_language.jsonl` 每条轨迹同时包含 `minimal_chain_id`、`minimal_chain`、`attempt_chain`、`stage_status`、`command_chains`、`command_chain_ids`。
3. `frame_language.jsonl` 每帧包含 `step_operation`，且 `step_operation` 不含 `Nx`/`1x`/`2x` 等重复次数标记。
4. `third_party/ada_manip/cfg/language_template.json` 不包含 `expand`、`n_min`、`n_max`、`expanded` 字段。
5. `language_expanded.json` 使用 `expanded_minimal_chains`，类型为 `List[List[str]]`。
6. `command_chains` 中每条链都必须可在任务级 `expanded_minimal_chains` 中找到同值链，且前缀匹配 `prefix`；`command_chain_ids` 必须与之逐项对应。
