# AdaManip Data Collection Language Schema (Final)

本文档定义数据采集阶段的语言标注存储规范。

本版本采用单一确定方案：外挂文件，不改 zarr 主结构。

## 1. 设计目标

1. 提供统一任务语言模板配置，结构化记录任务命令、操作集合、抽象步骤链。
2. 采集时按任务名读取模板，并在输出数据目录写入本次采集对应的语言文件。
3. 含 `Nx` 的链在采集后按实际轨迹统计进行离散展开，避免训练时动态解析。
4. 同时保存三层语言标注：任务级、轨迹级、帧级。

## 2. 采集主数据格式与 Episode 写入

### 2.1 主数据格式（zarr）

AdaManip 的主训练数据存储在 zarr 文件中，通常为 `demo_data.zip`。核心字段为：

1. `data/pcs`：每帧点云。
2. `data/env_state`：每帧低维状态。
3. `data/action`：每帧动作。
4. `meta/episode_ends`：每条 episode 的结束帧累计偏移。

`episode_ends` 是 episode 边界的唯一依据。例如 `episode_ends = [10, 25, 42]` 时：

1. episode 0 对应帧区间 `[0, 10)`。
2. episode 1 对应帧区间 `[10, 25)`。
3. episode 2 对应帧区间 `[25, 42)`。

因此主数组是按帧串接存储，但 episode 通过 `episode_ends` 明确隔离。

### 2.2 采集阶段 Episode 写入语义（当前实现）

以 microwave 采集实现为例，当前写入流程为：

1. 每轮 rollout 开始时，为每个并行环境创建一个 `Episode_Buffer`。
2. 每次执行 `process_data()` 时，向对应环境缓冲追加一帧 `(pc, env_state, action)`。
3. rollout 结束后，仅将成功环境对应的 `Episode_Buffer` 追加到全局 `Experience`。
4. 采集结束时，通过 `Experience.save()` 写出单个 zarr 文件。

含义：

1. 一个成功环境轨迹对应一个保存后的 episode。
2. 一轮外层 rollout 可产出多个保存 episode（因为 `num_envs` 并行，按成功环境分别落盘）。
3. 当前语言 sidecar 中 `episode_id` 也是按“成功轨迹”递增，与保存进 zarr 的 episode 一一对应。

## 3. 输出文件（外挂）

每个采集数据目录包含：

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

## 4. 为什么模板不包含 expand 属性

结论：模板层不放 `expand`、`n_min/n_max`、`expanded`。

原因：
1. 对部分任务，`N` 的有效范围依赖真实采集执行，不应在采集前硬编码。
2. 若模板预填错误范围，会污染数据并引入二次修订成本。
3. 模板用于“输入定义”，展开用于“输出事实”，两者应解耦。

因此采用两阶段：
1. 采集前读取模板（抽象链，允许 `Nx` 占位）。
2. 采集后根据实际轨迹统计生成 `language_expanded.json`（事实链，无 `Nx`）。

## 5. 统一任务语言模板格式

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

## 6. 采集后展开配置格式

`language_expanded.json` 记录本次任务采集实际出现的全部最优（无冗余步骤）离散链（全部轨迹的minimal_chain的集合）：

```json
{
  "schema_version": "v1",
  "generated_from": "language_template.json",
  "task": "bottle",
  "command": "打开瓶子",
  "operation_set": ["旋转瓶盖", "向上提起瓶盖"],
  "expanded_minimal_chains": [
    ["向上提起瓶盖"],
    ["1x旋转瓶盖", "向上提起瓶盖"],
    ["2x旋转瓶盖", "向上提起瓶盖"],
    ["3x旋转瓶盖", "向上提起瓶盖"]
  ],
  "attempt_chain_counts": [
    {
      "attempt_chain": ["向上提起瓶盖"],
      "count": 12
    },
    {
      "attempt_chain": ["1x旋转瓶盖", "向上提起瓶盖"],
      "count": 25
    },
    {
      "attempt_chain": ["1x旋转瓶盖", "向上提起瓶盖", "1x旋转瓶盖", "向上提起瓶盖"],
      "count": 7
    }
  ]
}
```

约定：`expanded_minimal_chains` 的索引即最短链 id（从 0 开始）；`generated_from` 用相对路径。
约定：`attempt_chain_counts` 统计本次数据中每种完整 `attempt_chain` 出现次数，计数口径按轨迹条数（不是帧数）。

## 7. 轨迹级语言标注（双链）

> 各 chain 字段的含义、计算流程、与训练 / 推理的关系详见 [`docs/design/chain_concepts.md`](chain_concepts.md)。本节只列 schema 字段定义。

`trajectory_language.jsonl` 记录本次任务重每条轨迹的语言标注。每条轨迹必须保存两类链：

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

## 8. 帧级语言标注（operation-only）

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

## 9. Bottle 任务示例（端到端）

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

## 10. `Nx` 任务的 N 范围由什么决定

涉及 `Nx` 占位的 4 个任务（`bottle` / `pen` / `pressure_cooker` / `coffee_maker`）的 `N` 不是直接配置出来的，而是采集阶段每条 successful trajectory 实际跑完后落出来的事实。

`minimal_chain` 中的 `Nx` 是 demo 跑完一条 trajectory 时 **整段累计**的旋转次数（`N_total = sum(K_i)`，`K_i` 是 attempt_chain 中第 `i` 段连续旋转的次数）。这一组 `N_total` 去重后即为 `expanded_minimal_chains` 中所有 `Nx 旋转... -> 向上提起...` 链的 N 值集合。详细的 chain 含义与计算流程见 [`docs/design/chain_concepts.md`](chain_concepts.md)。

观察到的 `N_total` 受以下参数共同约束：

### 10.1 下界：`try_range`（env 端）

`open_<task>.py` 的 `ada_policy` 在 `|dof| < self.env.try_range` 时强制返回 `r` / `o`。`N_total` 至少要让 dof 跨过 `try_range` 才有可能让某次抬升真的成功开盖，所以下界粗略为：

```
N_total_min ≈ ceil(try_range / 单步 dof 推进量)
```

`try_range` 在 env 的 `__init__` 写死，按任务取值（以及推导）：

| 任务 | `try_range` | 注释（env 源码原话） |
|---|---|---|
| `pen` | `0.99875` | `min random_range * open_stage_scale --> 2.35*0.5*0.85` |
| `bottle` | `0.99875` | `min random_range * open_stage_scale --> 2.35*0.5*0.85` |
| `pressure_cooker` | `0.35` | `min random_range * open_stage_scale --> 0.824*0.5*0.85` |
| `coffee_maker` | `0.34` | `min random_range * open_stage_scale --> 0.8*0.5*0.85` |

公式拆解：
- `min random_range = (1 - limit_random) × upper_limit_from_urdf`，对应 `cfg.env.asset.limit_random`（4 个任务都是 `0.5`）和 URDF 里这一关节的最大角度。
- `0.85` 是 demo 用的安全余量（`open_stage_scale`），写死在 env 的 `__init__` 里，不暴露到 cfg。

“单步 dof 推进量”随任务不同：
- `pen` / `bottle`：`r`/`o` 用 `quat_mul` 对夹爪施加 `±0.1305262 rad ≈ 15°` 的姿态增量，传到 cap dof 上一般 ≤ 15°，受抓取效率/打滑影响。
- `pressure_cooker` / `coffee_maker`：`r` 用 `pre_p[i] ± rotate_dir[i] * step_size`（`step_size = 0.035 m` ≈ 切向位移 3.5 cm），handle 半径决定每步换算成多少弧度的 dof 推进。

### 10.2 上界：`max_step`（manipulation 端）

`collect_manip_data` 的最外层 `for t in range(max_step):` 是硬上界，每条 trajectory 最多执行 `max_step` 个动作，因此 `N_total ≤ max_step - 1`（成功 trajectory 的最后一步必须留给 `z`）。

各任务 `max_step` 写死在 `manipulation/open_<task>.py` 里，按 `cfg.task.policy` 切换：

| 任务 | `policy="adaptive"` | `policy="succ"` |
|---|---|---|
| `pen` | `30` | `25` |
| `bottle` | `25` | `20` |
| `pressure_cooker` | `25` | `20` |
| `coffee_maker` | `20` | `15` |

> 实测：`pen` 在当前参数下 `N_total` 通常落在 `12~19` 区间。低端（≈12–14）来自 demo 一上来就连转再抬一次成功的情况；高端（≈17–19）来自 demo 多次试错（rotate→fail-lift→rotate→fail-lift→…→succ-lift），最后累计旋转次数比 `N_total_min` 多出几次。

### 10.3 中间：`ada_policy` 在 `|dof| ≥ try_range` 之后的随机继续

`ada_policy` 走过 `try_range` 后改为按经验先验采样：上一拍是 `z` 但 `open_bottle_stage` 仍为 False，则继续 `r`/`o`；上一拍是 `r`/`o`，以 `prob = 11/20` 切到 `z`，否则继续 `r`/`o`。这条分支让 `N_total` 在 `N_total_min` 之上还会再加 0~若干次旋转（来自重试），所以同一任务下 `expanded_minimal_chains` 通常是几个相近 `Nx` 值组成的离散集合。

### 10.4 想要采集到更宽 / 更窄的 N 分布该改哪里

| 想要的效果 | 改动点 |
|---|---|
| 让 `N_total_min` 更小 | env 端调小 `try_range`（直接改源码的字面量），或调大 `cfg.env.asset.limit_random`（让 URDF 上限随机区间更靠近 0）。 |
| 让 `N_total_min` 更大 | 反向，调大 `try_range` 或调小 `limit_random`。 |
| 让 `N_total` 分布更集中（少重试加成） | 把 `ada_policy` 走过 `try_range` 之后的 `prob = 11/20` 调高（越快切 `z` 越倾向首试就成功），同时加大单次 `z` 的位移幅度（`open_size`）让单次抬升更易顶开 cap。两者都会减少重试次数。 |
| 让 `N_total` 上界更大 | manipulation 端调大 `collect_manip_data` 里的 `max_step`（不要忘了 `cfg.env.horizon` 也要 ≥ `max_step`，否则 `action_chosen` 写超界）。 |
| 想要 `clock_wise == 0`（直接 `["向上提起..."]`）的 trajectory | `cfg.env.clockwise` 是新 env 中 `use_clockwise=True` 的概率（`pen`/`bottle` 默认 `0.5`、`coffee_maker` 是 `1.0` 即始终需要旋转、`pressure_cooker` 是 `0.0` 即从不旋转）。 |

cfg 里和这些参数相关的字段汇总：

```yaml
task:
  num_episode: 20      # rollout 次数；每次产出 ≤ num_envs 条 trajectory
  policy: adaptive     # adaptive / succ；决定 max_step
env:
  numEnvs: 10          # 并行环境数
  horizon: 35          # action_chosen 缓冲长度，必须 ≥ max_step
  clockwise: 0.5       # 单 env 抽到 clock_wise=1 的概率
  asset:
    limit_random: 0.5  # 关节随机区间宽度系数
```

`try_range`、`open_stage_scale=0.85`、`max_step` 是 demo 实现细节，目前没有暴露成 cfg；若要做超参扫，请修改对应的 `envs/open_<task>.py` 与 `manipulation/open_<task>.py`。

## 11. 最小验收标准

1. 数据目录中存在 `language_expanded.json`。
2. `trajectory_language.jsonl` 每条轨迹同时包含 `minimal_chain_id`、`minimal_chain`、`attempt_chain`、`stage_status`、`command_chains`、`command_chain_ids`。
3. `frame_language.jsonl` 每帧包含 `step_operation`，且 `step_operation` 不含 `Nx`/`1x`/`2x` 等重复次数标记。
4. `third_party/ada_manip/cfg/language_template.json` 不包含 `expand`、`n_min`、`n_max`、`expanded` 字段。
5. `language_expanded.json` 使用 `expanded_minimal_chains`，类型为 `List[List[str]]`。
6. `command_chains` 中每条链都必须可在任务级 `expanded_minimal_chains` 中找到同值链，且前缀匹配 `prefix`；`command_chain_ids` 必须与之逐项对应。
7. `language_expanded.json` 必须包含 `attempt_chain_counts`，且其 `count` 总和等于 `trajectory_language.jsonl` 中轨迹总数。

语言条件向量的生成与存储规范已迁移到 `docs/design/data_preprocess.md`，本文件仅覆盖采集阶段内容。
