# Chain 概念清单

本文档把 `language_expanded.json` / `trajectory_language.jsonl` / `frame_language.jsonl` 中出现的各种"链"集中讲清楚，避免混淆。`docs/design/data_collection.md` 是 schema 字段规约，本文档是 chain 字段语义的解释及具体计算规则。

## 1. chain 字段一览

| 字段 | 出现位置 | 含义 | 来源 |
|---|---|---|---|
| `minimal_chain` | trajectory 级 | 把 `attempt_chain` 聚合成"如果一次跑通最短长这样"的链（丢失败 stage + 合并同操作的 Nx）。**仍是 attempt 推导，不看 env 内部状态**。 | 纯 attempt 推导 |
| `ground_truth_chain` | trajectory 级 | **上帝视角**最优链——基于本 env 的内部状态（cw 等），不看 demo 是否走弯路。 | env state 推导 |
| `attempt_chain` | trajectory 级 | demo 真实执行的完整动作序列（含失败重试）。 | 直接从 demo 状态机记录 |
| `command_chains` | trajectory 级 | 训练时可作为语言条件采样的候选 chain 集合（List[List[str]]）；按 schema prefix 规则在 bank 上做匹配。 | attempt + bank |
| `expanded_minimal_chains` | 任务级 | 任务级 chain id 索引——`Nx` 任务上**强行展开为 1..N_max 全集**，确保 schema 的 prefix 匹配总能命中（见 §5）。 | bank 构造 |
| `expanded_actual_minimal_chains` | 任务级 | 本次采集真正出现过的 minimal_chain 去重后的列表，是 `expanded_minimal_chains` 的子集（`expanded_actual_minimal_chains ⊆ expanded_minimal_chains`）。 | 实际观测集合 |

> `minimal_chain` 与 `ground_truth_chain` 的关键区别：前者是"demo 跑的过程聚合后最短能长成什么样"（看 attempt_chain，不知 env 内部）；后者是"已知 env 状态的话最短长什么样"（看 env 状态，不看 demo 是否绕弯）。demo 一次性成功且没多余动作时两者相同，否则可能不同——见 §2 末尾的对照例子和 §3。

## 2. `minimal_chain` —— 从 `attempt_chain` 聚合得到的最简链

`minimal_chain` 严格从 `attempt_chain` + `stage_status` 推导，不查 env 状态。规则两步：

1. **丢失败 stage**：把 `stage_status` 为 `False` 的 stage 全部去掉（这些 stage 是 demo 走过的弯路，没把任务推进）。
2. **合并相邻同操作 stage**：
   - 形如 `Nx{op}` 的重复 stage：相邻同 op 时，N 累加（`1x旋转 + 1x旋转 → 2x旋转`）。
   - 非重复 stage：相邻同字符串去重（`[拉门, 拉门] → [拉门]`，实际几乎不出现）。

得到的就是"如果 demo 一次性把这些操作连续跑出来（不试错、不重复），最短能这么写"的链，因此叫 minimal_chain。

### 2.1 例 1（bottle 失败重试，data_collection.md §7 那个例子的正确版本）

```json
"attempt_chain":  ["1x旋转瓶盖", "向上提起瓶盖", "1x旋转瓶盖", "向上提起瓶盖"],
"stage_status":   [true,         false,         true,         true]
// 丢索引 1 的失败 lift → ["1x旋转瓶盖", "1x旋转瓶盖", "向上提起瓶盖"]
// 合并相邻的两个 Nx 旋转瓶盖 → ["2x旋转瓶盖", "向上提起瓶盖"]
// minimal_chain = ["2x旋转瓶盖", "向上提起瓶盖"]
```

> `data_collection.md §7` 的示例 JSON 把 minimal 写成了 `1x`，那是 doc 笔误：实际跑过 1+1=2 次旋转，2x 才足以让 cap 真正松掉、最后一次 lift 才会成功。规则以本文档及 `extract_minimal_chain_from_attempt` 实现为准。

### 2.2 例 2（pen 多次重试）

```json
"attempt_chain":  ["9x旋转笔盖", "向上提起笔盖", "1x旋转笔盖", "向上提起笔盖", "5x旋转笔盖", "向上提起笔盖"],
"stage_status":   [true,         false,         true,         false,         true,         true]
// 丢两次失败 lift → ["9x旋转笔盖", "1x旋转笔盖", "5x旋转笔盖", "向上提起笔盖"]
// 合并 → ["15x旋转笔盖", "向上提起笔盖"]
```

### 2.3 例 3（microwave start_with_pull=False, cw=0）

```json
"attempt_chain":  ["按按钮", "拉门"],
"stage_status":   [true,    true]
// 没有 False、没有相邻同 op → minimal_chain == attempt_chain
// minimal_chain = ["按按钮", "拉门"]
```

注意：env 是 cw=0 时按按钮其实多余，但 attempt 视角看不出来（demo 没失败 stage 可丢，相邻 op 不同也不可合并），所以 `minimal_chain` 保留了"按按钮"。要用上帝视角（cw 已知）判定按按钮多余、得到 `["拉门"]`，那是 `ground_truth_chain` 的活，见 §3。

### 2.4 例 4（pen 一次成功，无失败 + 无可合并）

```json
"attempt_chain":  ["12x旋转笔盖", "向上提起笔盖"],
"stage_status":   [true,          true]
// minimal_chain = attempt_chain == ["12x旋转笔盖", "向上提起笔盖"]
```

### 2.5 实现

`manipulation/language_chain_utils.py::extract_minimal_chain_from_attempt(attempt_chain, stage_status) -> List[str]`，`base_manipulation.py::collect_episode_end` 直接调用。更多边界例子见 `tests/show_language_chain_reasoning_examples.py`。

## 3. `ground_truth_chain` —— 上帝视角的最优链

`ground_truth_chain` 不看 demo 跑了什么，只看 env 的内部状态（cw、限位等），给出"完成这个 env 需要的最短 / 最优操作序列"。

### 3.1 ground_truth 必须是 env state 的纯函数

核心原则：**对同一个 env state，ground_truth 必须是唯一的**，与 demo 跑出来怎样、跑了几个 episode 都无关。否则同一 env 的不同 episode 会算出不同 ground_truth，破坏"上帝视角"的语义。

各任务 cw=0 / cw=1 → ground_truth 的取值：

| 任务 | cw=0 ground_truth | cw=1 ground_truth | 来源 |
|---|---|---|---|
| microwave | `["拉门"]` | `["按按钮", "拉门"]` | base 默认 `canonical_minimal_chain_for_state` |
| door / safe / window / lamp | 任务自定 | 任务自定 | base 默认 |
| pen / bottle | `["向上提起笔盖"]` / `["向上提起瓶盖"]` | `["{N_min}x旋转笔盖", "向上提起笔盖"]` / `["{N_min}x旋转瓶盖", "向上提起瓶盖"]` | task override（env 物理观测） |
| pressure_cooker / coffee_maker | `["向上提起把手"]` / `["拉动手柄"]` | `["{N_min}x旋转把手", "向上提起把手"]` / `["{N_min}x旋转手柄", "拉动手柄"]` | task override（env 物理观测） |

> **关于 `Nx` 任务 cw=1 的具体 N_min**：pen / bottle / pc / cm 的 cap 在 cw=1 时锁着，需要 rotate；具体 N_min 由 env 物理决定（cap 的 `random_lower/upper` 限位 + 单步 dof 推进效率）。从单一 env state snapshot 不能解析地反推，但是**在 demo 跑的过程中是可观测的**——env 自己维护一个 `open_bottle_stage[env_id]` flag，每一拍根据 cap dof 是否跨过解锁阈值（`|two_dof[0]| ≥ 0.85 × (upper - lower)`）更新；demo 只要插桩记录"这个 flag 第一次从 False 翻到 True 那一拍累计转了几次"，就是这个 env 的 N_min。env 物理决定 N_min，所以**同一 env 跨不同 episode 仍然得到同样的 N_min**（即便 demo 因随机选择转得更多，N_min 不变）。
>
> **关于 cw=0 / `open_bottle_stage` 一开始就是 True 的 env**：N_min = 0、ground_truth = `["{lift_op}"]`。如果 cw=0 的 env 一开始 `open_bottle_stage` 仍是 False（cap 也需要先转才能松），N_min > 0、ground_truth = `["{N_min}x{rotate}", "{lift}"]`，与 cw=1 同形式。也就是说 **cw 不直接决定结构，env 物理决定**——这与 pen 等任务 cw 只决定旋转方向、不决定"是否需要旋转"的真实情况一致。

### 3.2 实现

钩子 `ground_truth_chain_for_collect(env_id, episode_state) -> List[str]`：

- base 默认：返回 `canonical_minimal_chain_for_state(state)`（适用于 microwave / door / safe / window / lamp 这 5 个非 Nx 任务）。
- pen / bottle / pc / cm 各自 override：从 episode-级 instance state `_<task>_intrinsic_n[env_id]`（在 `collect_manip_data` 里维护）读出 N_min，套上对应 task 的 rotate/lift op 拼成 chain。

**N_min 怎么记的**（在 `collect_manip_data` 内层循环里）：

```python
# 每 episode 开头初始化
self._<task>_intrinsic_n = [None] * num_envs   # 还没看到 open_bottle_stage 翻转
self._<task>_cum_rot     = [0]    * num_envs   # 累计旋转次数

# 抓握完、进 manip 主循环之前先检查一遍（兼容 init 即 True 的 env）
for env_id in range(num_envs):
    if bool(self.env.open_bottle_stage[env_id].item()) and \
            self._<task>_intrinsic_n[env_id] is None:
        self._<task>_intrinsic_n[env_id] = 0

# 主循环每一拍 env.step 之后
for env_id in range(num_envs):
    if res_per_env[env_id] in ("r", "o"):
        self._<task>_cum_rot[env_id] += 1
    if bool(self.env.open_bottle_stage[env_id].item()) and \
            self._<task>_intrinsic_n[env_id] is None:
        # 第一次翻转：env 物理刚刚判定"够松了"，把当前累计转数定为 N_min
        self._<task>_intrinsic_n[env_id] = int(self._<task>_cum_rot[env_id])
```

> **N_min vs minimal_chain 的累计 N**：minimal_chain 取 demo 真做了的累计 rot 数（>=N_min，因为 demo 走过 try_range 后还可能因 ada_policy 的随机继续多转几下）；ground_truth 取的是 env 物理的 N_min（精确边界值）。在 demo 一过 try_range 就立刻切 lift 的"幸运 episode" 上两者相等；多转了几下的 episode 上 minimal_chain 会比 ground_truth 大 1~几次。

### 3.3 用途

- 离线 asker 把视频/轨迹送进 LLM 让其推断"这条 trajectory 应该用哪条链"，asker 输出与 `ground_truth_chain` 严格相等比对，得到对错统计。
- 评估阶段（adaptive eval state machine）的 chain id 集合通常以 ground_truth 为锚点。
- 训练若想强制 cw 一致性，可以把 `ground_truth_chain` 当类别监督；默认走 `command_chain_ids` 的 dataloader 不读这个字段。

### 3.4 与 `minimal_chain` 的对照

记 `K` = demo 累计旋转次数（minimal_chain 的 N），`N_min` = env 物理 N_min（ground_truth 的 N，由 `open_bottle_stage` 翻转点观测）。`K ≥ N_min` 始终成立——demo 不会比 env 物理需要的更早能成功。

| 场景 | minimal_chain（attempt 聚合）| ground_truth_chain（env state 纯函数）| 是否一致 |
|---|---|---|---|
| **microwave** cw=0, demo 直接拉门 | `["拉门"]` | `["拉门"]` | ✓ |
| **microwave** cw=0, demo 多按了按钮 | `["按按钮", "拉门"]` | `["拉门"]` | ✗ 结构不同 |
| **microwave** cw=1, demo 顺利按按钮+拉门 | `["按按钮", "拉门"]` | `["按按钮", "拉门"]` | ✓ |
| **microwave** cw=1, demo 先误拉门、再按按钮+拉门 | `["按按钮", "拉门"]`（丢首段失败 lift 后聚合）| `["按按钮", "拉门"]` | ✓ |
| **pen / bottle / pc / cm** demo 一进入 manip loop 就立刻满足 `open_bottle_stage` | `["{K}x旋转", "{lift}"]` 或 `["{lift}"]` | `["{lift}"]` | demo 无多余 → ✓；demo 有多余 rotate → ✗ |
| **pen / bottle / pc / cm** demo 刚过 try_range 就切 lift | `["{N_min}x旋转", "{lift}"]` | `["{N_min}x旋转", "{lift}"]` | ✓ |
| **pen / bottle / pc / cm** demo 过了 try_range 还多转了 ΔK 次 | `["{N_min+ΔK}x旋转", "{lift}"]` | `["{N_min}x旋转", "{lift}"]` | ✗ K > N_min |
| **pen / bottle / pc / cm** demo 多轮重试 (rotate→fail-lift→rotate→…→succ-lift) | `["{K_total}x旋转", "{lift}"]`（聚合后总 K_total） | `["{N_min}x旋转", "{lift}"]` | ✗ K_total ≥ N_min |

> 不一致只有两种情况：
> - **结构不同**：env 视角某些 stage 完全多余（microwave cw=0 多按按钮、pen 等任务 init 即可 lift 的 env 上 demo 多转了 rotate）。
> - **N 不同**：env 视角 N=N_min（精确边界），demo 因为 `ada_policy` 的随机继续在 `|dof| > try_range` 之后又多转几下，K > N_min。
>
> 想严格按 env 物理监督走 `ground_truth_chain`；想跟随 demo 实际行为（包括其多余动作）走 `minimal_chain`。两者**对同一 trajectory** 的语义都是定义清晰的；ground_truth 还满足"同一 env 跨多次 episode 不变"的额外性质。

## 3. `attempt_chain` —— demo 真实执行的链

`attempt_chain` 按真实顺序逐段记录 demo 的动作。一段（stage）的判断：连续相同操作合并成一个 stage：

- 连续 `K` 次旋转 → 一个 stage `"{K}x旋转..."`。
- 一次抬升（lift / 拉门）→ 一个 stage `"向上提起..."`。
- stage 之间发生切换（rotate ↔ lift）就开新 stage。

`stage_status` 是与 `attempt_chain` 等长的 bool 列表：

- 旋转 stage 永远 `True`（每一次旋转都"成功推进了 dof"）。
- lift stage：`False` 表示这次抬升没把锁顶开（cap 还黏着）；`True` 表示真把任务完成了。
- 成功 trajectory 的最后一个 stage 必须是 lift 且 `True`。

### 例（pen 真实数据）

```json
"attempt_chain": ["9x旋转笔盖", "向上提起笔盖", "1x旋转笔盖", "向上提起笔盖", "5x旋转笔盖", "向上提起笔盖"],
"stage_status": [true, false, true, false, true, true]
```

读法：旋转 9 次 → 试抬（失败）→ 再转 1 次 → 试抬（失败）→ 再转 5 次 → 试抬（成功）。
对应的 `minimal_chain = ["{9+1+5}x旋转笔盖", "向上提起笔盖"] = ["15x旋转笔盖", "向上提起笔盖"]`（`N_total = 15`）。

实现位置：`manipulation/open_<task>.py::concrete_attempt_chain_for_collect`，依赖在 `collect_manip_data` 里维护的 `_<task>_attempt_chains` / `_<task>_stage_statuses`。每一拍 demo 决定 op 之后，op 切换时把上一段 `_flush` 出去（决定其 `stage_status`：旋转→True，lift→进入下一拍才知道是否是最后成功的一次→暂记 False，最后成功的那次抬升在 done_flag 检查时再次 `_flush(success=True)` 显式覆写）。

## 4. `command_chains` —— 训练时按这个采样语言条件

`command_chains` 是 dataloader 在 batch 时为每条 trajectory 选语言 embedding 时的**候选集**——dataloader 从 `command_chains` 中随机挑一条，用其 chain id 取 `language_embedding_dict.json` 的对应 embedding 作为 condition。

### 4.1 计算规则

严格按 `data_collection.md §7`：

1. 找 `attempt_chain` 中第一个 `stage_status == False` 的位置 `k`，无失败则 `k = len-1`。
2. `prefix = attempt_chain[:k+1]`。
3. 在 `expanded_minimal_chains` 中筛选所有以 `prefix` 为起始的链，组成 `command_chains` 与对应 `command_chain_ids`。

之所以 prefix 总能命中，是因为 §5 中描述的 **`expanded_minimal_chains` 已被强行展开为 1..N_max 全集**——例如 `prefix = ["9x旋转笔盖", "向上提起笔盖"]`，bank 里就一定有对应 N=9 的那条；`prefix = ["向上提起笔盖"]`，bank 里就一定有 cw=0 那条。schema "command_chains 必须非空"这个硬约束因此自然成立，没有 fallback 路径。

实际计算位置：`base_manipulation.py::collect_finalize`（在 bank 拓展完成之后再次调用 `match_command_chains`，所以即使 `collect_episode_end` 阶段 bank 还没 ready，最终结果也是按全集匹配的）。

### 4.2 三个常见 case（用 pen 举例，bank 已展开为 1..19）

| 场景 | attempt_chain[0] | first_fail | prefix | command_chain_ids |
|---|---|---|---|---|
| cw=0 env，第一拍 demo 直接抬升、成功 | `"向上提起笔盖"` | None（无失败） | `["向上提起笔盖"]` | `[0]` → `["向上提起笔盖"]` |
| cw=1 env，第一拍 demo 抬升、失败，后面再旋转直到成功 | `"向上提起笔盖"` | 0 | `["向上提起笔盖"]` | `[0]` → `["向上提起笔盖"]`（与本 env 的 minimal_chain 不一致；schema 的"命令一致性优先"决定的：demo 第一拍下达的就是 lift-only 命令） |
| cw=1 env，第一拍 demo 转 K 次再抬升失败、继续重试 | `"{K}x旋转笔盖"` | 1（第一次抬升失败） | `["{K}x旋转笔盖", "向上提起笔盖"]` | `[K]` → `["{K}x旋转笔盖", "向上提起笔盖"]` |

> case 2 的 `command_chains` 是 cw=0 链而本 env 实际 cw=1。schema 把这种 trajectory 视为 "demo 第一次下达的命令是 cw=0 那条，后来偏离了"。训练若对 cw 一致性敏感，dataloader 可以走 `minimal_chain_id` 而不是 `command_chain_ids`，绕过这个偏置。

## 5. 两个 bank：`expanded_minimal_chains` 与 `expanded_actual_minimal_chains`

`language_expanded.json` 同时落两份 bank：

### 5.1 `expanded_actual_minimal_chains`：本次真实出现过的链（事实集）

去重后按出现顺序写入，是 `expanded_minimal_chains` 的子集：

```jsonc
"expanded_actual_minimal_chains": [
  ["向上提起笔盖"],
  ["17x旋转笔盖", "向上提起笔盖"],
  ["15x旋转笔盖", "向上提起笔盖"],
  ["12x旋转笔盖", "向上提起笔盖"],
  ["19x旋转笔盖", "向上提起笔盖"],
  ["13x旋转笔盖", "向上提起笔盖"]
]
```

它的用途：人工分析 / 数据 EDA / 后续 round 累计统计（看每次采集到了哪些 N），与 chain id 索引无关——`trajectory_language.jsonl` 里的 id 不指向这个数组。

### 5.2 `expanded_minimal_chains`：强行展开到 1..N_max 的索引集（chain id 真正的索引表）

为了让 §4 的 prefix 规则在 demo 走出"中间 N"前缀时也总能命中，bank 在 `collect_finalize` 里被强行展开：

- 单元素或非 `Nx` 模板的 chain（如 `["向上提起笔盖"]`、`["按按钮", "拉门"]`）原样保留。
- 每个 `(rotation_op_root, lift_op)` 对（例如 `("旋转笔盖", "向上提起笔盖")`），按观察到的最大 N 展开成 `1x..N_maxx`。

pen 例子（actual N_max = 19）：

```jsonc
"expanded_minimal_chains": [
  ["向上提起笔盖"],                       // [0]：cw=0 的 lift-only 链
  ["1x旋转笔盖", "向上提起笔盖"],         // [1]：实际可能没出现，但 schema 规则需要
  ["2x旋转笔盖", "向上提起笔盖"],         // [2]
  ...
  ["12x旋转笔盖", "向上提起笔盖"],        // [12]：本次出现过
  ...
  ["19x旋转笔盖", "向上提起笔盖"]         // [19]：本次最大值
]
```

下标即 chain id，`trajectory_language.jsonl` 中所有 `minimal_chain_id` / `command_chain_ids` 都引用这个数组的下标。

### 5.3 `_build_full_minimal_chain_bank` 的展开规则

实现位置：`base_manipulation.py::BaseManipulation._build_full_minimal_chain_bank(observed_chains)`。规则：

1. 遍历 `observed_chains`：
   - 长度 == 2 且第 1 元素匹配 `^(\d+)x(.+)$` → 提取 `(N, op_root, lift)`，计入 `nx_max_by_op[(op_root, lift)] = max(N)`，按首次出现顺序登记 `(op_root, lift)`。
   - 否则原样并入 `non_nx_chains`（保留观察顺序，去重）。
2. 输出：`non_nx_chains + [["{i}x{op_root}", lift] for (op_root, lift) in op_order for i in 1..max_N]`。

这个默认实现适配 4 个 `Nx` 任务（pen / bottle / pc / cm）和 5 个非 `Nx` 任务（microwave / door / safe / window / lamp）：后者的所有 chain 都走 non-Nx 分支，bank 直接等于 actual。

### 5.4 chain id 的最终映射时机

`collect_finalize` 顺序：

1. 过滤掉模板里残留的、含字面 `Nx` 的 abstract chain → 得到 `expanded_actual_minimal_chains`。
2. `_build_full_minimal_chain_bank` 展开 → 得到 `expanded_minimal_chains`。
3. 对每条 trajectory_record 重新求 `minimal_chain_id`（在全集 bank 中查 `minimal_chain` 的 index）和 `command_chain_ids`（重新跑 `match_command_chains`）。
4. 把两个 bank、所有 trajectory_records 写到 `language_expanded.json` / `trajectory_language.jsonl`。

之所以选择在 finalize 阶段一次性映射，是因为 `collect_episode_end` 触发时 bank 还在动态扩展（每 epoch 末尾才追加新观察到的 chain），早期算出的 chain id 可能在 finalize 阶段无效；finalize 末尾再算一次，所有 trajectory 看到的是同一个最终 bank。

### 5.5 为什么 cw=0 的 `["向上提起笔盖"]` 会出现，即使本次没有 cw=0 的成功 trajectory

bank 在 episode 开始前就用 `cfg/language_template.json` 的 `minimal_chains` 字段初始化了一份 abstract 版本（`["向上提起笔盖", "Nx旋转笔盖 -> 向上提起笔盖"]` → `[["向上提起笔盖"], ["Nx旋转笔盖", "向上提起笔盖"]]`）。`collect_finalize` 过滤的是含 `Nx` 字面值的链，`["向上提起笔盖"]` 不含 `Nx` 所以会保留到 `expanded_actual_minimal_chains` 中——即使本次采集所有 env 都是 cw=1。然后再被全集展开带到 `expanded_minimal_chains[0]`。

> 这条规则服务于 §4 case 2 的 `command_chains`：bank 里始终保留模板提供的所有"无 Nx"的 baseline chain，让 schema 的 prefix 规则在 demo 第一拍误抬时仍能匹到 `["向上提起笔盖"]`。

## 6. `frame_language.jsonl` 与 chain 的关系（提一下）

`frame_language.jsonl` 不存 chain 而是存逐帧的 `step_operation`（属于 `operation_set`，例如 `"旋转笔盖"` / `"向上提起笔盖"`，**不带 N**）。`step_index` 指向**该 env 当时所在 attempt_chain stage 的下标**：第一段 rotate→step_index=0，第一次 lift→step_index=1，第二段 rotate→step_index=2 ……

每条 trajectory 在 `trajectory_language.jsonl` 里的 `frame_range = [start, end)` 标出它在 `frame_language.jsonl` 列表中的切片位置。

## 7. 结合实际 pen 数据的端到端例子

某条 cw=1 的 pen trajectory，demo 跑出来：

```
真实动作流: r r r r r r r r r z r z r z r z r r r r r z
合并 stage:    9x旋转  z  1x z  1x z  1x z   5x       z
              (T)    (F)(T)(F)(T)(F)(T)(F)  (T)     (T)
```

- `attempt_chain = ["9x旋转笔盖", "向上提起笔盖", "1x旋转笔盖", "向上提起笔盖", "1x旋转笔盖", "向上提起笔盖", "1x旋转笔盖", "向上提起笔盖", "5x旋转笔盖", "向上提起笔盖"]`
- `stage_status = [True, False, True, False, True, False, True, False, True, True]`
- 聚合：丢失败 lift（索引 1, 3, 5, 7）→ `["9x旋转笔盖", "1x旋转笔盖", "1x旋转笔盖", "1x旋转笔盖", "5x旋转笔盖", "向上提起笔盖"]` → 合并相邻 Nx → `["17x旋转笔盖", "向上提起笔盖"]`，所以 `minimal_chain = ["17x旋转笔盖", "向上提起笔盖"]`
- env 物理 N_min 通过 `open_bottle_stage` 翻转点观测：假设这个 env 在累计转到 15 次时 flag 翻转 → `ground_truth_chain = ["15x旋转笔盖", "向上提起笔盖"]`。注意 demo 总共转了 17 次（minimal_chain 的 K=17），但 env 物理上 15 次就够了；多出的 2 次是 ada_policy 在 `|dof| > try_range` 之后随机继续的副产品，不进入 ground_truth。**同一 env 跨不同 episode 始终得到 N_min=15**，与每次 demo 实际转了多少次无关。
- bank 全集（`expanded_minimal_chains`）展开成 1..N_max（N_max 由所有 attempt_chain 中出现的最大 N 决定，本例 = 9）。`minimal_chain = ["17x..."]` 在下标 17 → `minimal_chain_id = 17`（注意 N_max 至少 17 才装得下；实际 N_max 取 max(所有 trajectory 出现的 N) 与 max(所有 minimal_chain 的 N) 的并集）
- `prefix = attempt_chain[:2] = ["9x旋转笔盖", "向上提起笔盖"]`，bank 全集中下标 9 即此 chain → `command_chains = [["9x旋转笔盖", "向上提起笔盖"]]`、`command_chain_ids = [9]`
- `expanded_actual_minimal_chains` 包含 `["17x..."]`，但不包含 `["9x..."]`（没有 trajectory 的 minimal_chain 是 9x）。bank 强行展开后，命中 prefix 的 9x 也在里面 → `command_chains` 永远非空、不走 fallback
- `frame_records[frame_range]`：每一帧的 `step_index` 在 0~9 之间游走，`step_operation` 在 `"旋转笔盖"` / `"向上提起笔盖"` 之间切换

> 总结这一条 trajectory 的 4 个 chain：
> - `attempt_chain`：完整 10 段过程；
> - `minimal_chain`：`["17x旋转笔盖", "向上提起笔盖"]` —— attempt 聚合后的最简形式（丢 4 次失败 lift + 合并 4 段 rotate，K=17）；
> - `ground_truth_chain`：`["15x旋转笔盖", "向上提起笔盖"]` —— env 物理 N_min=15（demo 跑到累计 15 次时 `open_bottle_stage` 翻转为 True；但 ada_policy 多转了 2 次才切到 lift）。**同一 env 跨多个 episode 都是 N_min=15**，这是 env state 的物理属性，与 demo 走多远无关；
> - `command_chains`：`[["9x旋转笔盖", "向上提起笔盖"]]` —— demo 第一次下达的命令（"先转 9 次再抬"），到第一次失败为止的命令片段。

## 8. 不同 chain 的训练 / 推理用法

| 环节 | 用谁 | 怎么用 |
|---|---|---|
| 训练阶段语言条件 | `command_chain_ids[i]` 之一（dataloader 随机采样） | 取 `language_embedding_dict.json["expanded_minimal_chains"][cid]` 当条件向量（embedding 表是按全集 bank 编码的，cid 直接对得上） |
| 训练阶段类别监督（attempt 视角） | `minimal_chain_id` | 把"demo 真实成功路径的最简形式"作为分类标签 |
| 训练阶段类别监督（env 视角） | `ground_truth_chain` | 上帝视角的最优；对 cw 一致性要求高的训练或 contrastive anchor 用这个 |
| 评估阶段（adaptive）的 chain 选择 | 全集 bank 的 chain id | adaptive eval 在全集 bank 上做 state machine（locked / tried / sweep）；候选空间是 `expanded_minimal_chains`，不是只见过的那几个 |
| 离线 asker 评判 | asker 输出的 chain | 与 `ground_truth_chain` 做严格相等比对得到对错；asker 是从视频/轨迹里推断"这条 trajectory 应当走什么链"，与 ground_truth 对齐才算正确 |
| 数据 EDA / 跨 round 累计统计 | `expanded_actual_minimal_chains` | 看每次采集真正出现了哪些 minimal_chain，做闭环优化时可用 |

## 9. 参考代码位置

- chain 计算与 sidecar 写入：`manipulation/base_manipulation.py::collect_episode_end` / `collect_finalize` / `save_language_sidecars`
- minimal_chain 提取（suffix-after-last-fail）：直接 inline 在 `collect_episode_end` 里
- 任务级钩子：
  - `concrete_attempt_chain_for_collect(env_id, state)`：返回 `(attempt_chain, stage_status)`。pen / bottle / pc / cm / microwave 各自 override；其他任务用 base 默认（`(canonical, [True]*n)`）
  - `ground_truth_chain_for_collect(env_id, state)`：返回上帝视角最优。pen / bottle / pc / cm 各自 override（累计 N）；其他任务用 base 默认（`canonical_minimal_chain_for_state`）
  - `canonical_minimal_chain_for_state(state)`：返回该 cw 状态下任务定义的"标准答案" chain。所有任务都 override
- prefix 匹配：`manipulation/base_manipulation.py::match_command_chains`
- bank 强行展开（1..N_max）+ chain id 重映射 + abstract Nx 过滤：`manipulation/base_manipulation.py::_build_full_minimal_chain_bank` + `collect_finalize`
- 训练时按 chain id 取 embedding：`docs/design/data_preprocess.md` + `try_to_remember/...` 中的 dataloader
