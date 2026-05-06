# Chain 工具函数参考

本文档是 chain 相关工具函数的**实现细节参考**。它从 `docs/inference.md` / `docs/design/chain_concepts.md` 中抽出来，让那两份概念性文档保持轻量、只在需要的地方引用本文档。

涵盖范围：

- `manipulation/language_chain_utils.py`：抽象 chain 推理（不读 env 状态）。
- `manipulation/base_manipulation.py`：与 chain / ground_truth 相关的钩子和共享 helpers，加上日志辅助。

约定：除特别说明外，"chain" 指 `List[str]`，每个 element 是一个 stage（可能是原子操作 `"拉门"`、形如 `"3x旋转笔盖"` 的重复 stage、或 template-only 的 abstract stage `"Nx旋转笔盖"`）。

---

## 1. `language_chain_utils.py`

**纯函数模块**，输入是 chain 字符串列表 + bank（`expanded_minimal_chains`），不读 env / 物理状态。所有 `Nx` / `1x..` 解析靠正则 `^\s*(\d+)x(.+?)\s*$`（即 `_REPEAT_STAGE_RE`）；不匹配的 stage（如 `"拉门"` / `"按按钮"` / 模板里 literal `"Nx旋转笔盖"`）按整体字符串处理。

### 1.1 `normalize_chain(chain) -> List[str]`

把每个 stage `str.strip()`，丢掉 strip 后为空的 stage。所有其他公开函数对入参先过这一步、然后只用规范化后的副本。

### 1.2 `expand_stage_to_atomic(stage) -> List[str]`

把单个 stage 展开成原子操作列表：

- 形如 `Nx{op}` 的具体重复 stage（`N` 是正整数）→ `[op] * N`。
- 不匹配 `Nx` 模式的 stage → 原样保留为单元素列表 `[stage]`。

实现细节：用 `_REPEAT_STAGE_RE` 解析；`N <= 0` 抛 `ValueError`；`op` 部分 `strip()` 后若为空也抛 `ValueError`。

### 1.3 `expand_chain_to_atomic(chain) -> List[str]`

对 `normalize_chain(chain)` 的每个 stage 套 `expand_stage_to_atomic`，得到所有原子操作的扁平列表。

例：`["2x旋转瓶盖", "向上提起瓶盖"]` → `["旋转瓶盖", "旋转瓶盖", "向上提起瓶盖"]`。

### 1.4 `is_subsequence(needle, haystack) -> bool`

经典子序列检查（顺序对齐 + 不要求连续）。空 needle 永远返回 `True`。`O(len(haystack))`。

### 1.5 `chain_satisfies_ground_truth(language_chain, ground_truth_chain) -> bool`

`is_subsequence(expand_chain_to_atomic(ground_truth_chain), expand_chain_to_atomic(language_chain))`。

语义："如果 policy 严格按 `language_chain` 执行（无失误），是否能覆盖 `ground_truth_chain` 要求的所有原子操作"。在原子层面比较是为了让 `2x旋转` 满足 `1x旋转` 的需求；同时保留 stage 字面值，避免误判 `顺时针旋转` 与 `逆时针旋转` 等异向操作。

### 1.6 `extract_minimal_chain_from_attempt(attempt_chain, stage_status) -> List[str]`（**新增**）

把 `attempt_chain` 聚合成"如果一次跑通最短长这样"的 minimal_chain，**纯 attempt 推导，不查 env 状态**。

实现两步：

1. **丢失败 stage**：根据 `stage_status` 把对应 `False` 的 stage 全部去掉（demo 走过的弯路、未推进任务的部分）。
2. **合并相邻同操作 stage**：
   - 形如 `Nx{op}` 的重复 stage：相邻同 `op` 时 N 累加，例如 `1x旋转 + 1x旋转 → 2x旋转`。
   - 非重复 stage：相邻同字符串去重，例如 `[拉门, 拉门] → [拉门]`（实际很少出现）。

入参长度不一致抛 `ValueError`；`Nx` 段中的 N ≤ 0 也抛 `ValueError`（防御）。

例：

| 输入 attempt | 输入 stage_status | 输出 minimal |
|---|---|---|
| `[拉门]` | `[True]` | `[拉门]` |
| `[拉门, 按按钮, 拉门]` | `[False, True, True]` | `[按按钮, 拉门]` |
| `[1x旋转瓶盖, 向上提起瓶盖, 1x旋转瓶盖, 向上提起瓶盖]` | `[True, False, True, True]` | `[2x旋转瓶盖, 向上提起瓶盖]` |
| `[9x旋转笔盖, 向上提起笔盖, 1x旋转笔盖, 向上提起笔盖, 5x旋转笔盖, 向上提起笔盖]` | `[True, False, True, False, True, True]` | `[15x旋转笔盖, 向上提起笔盖]` |

完整示例脚本：[`tests/show_language_chain_reasoning_examples.py`](../../tests/show_language_chain_reasoning_examples.py)。

### 1.7 `infer_attempt_chain(language_chain, ground_truth_chain) -> List[str]`

抽象推理"如果 policy 严格按 `language_chain` 执行、且 ground truth 是 `ground_truth_chain`，会观察到什么完整 attempt"：

1. 先 `normalize_chain` 两个入参。
2. 临时把 stages 展开到原子层（用 `expand_chain_to_atomic`）。
3. 用 `is_subsequence` 检查 `ground_truth_chain` 的原子序列是不是 `language_chain` 的有序子序列。这里用子序列（不是前缀）：更长的语言条件可能涵盖真实最小链所有动作；同时区分 `顺时针旋转` 与 `逆时针旋转` 等异向操作。
4. 子序列检查通过 → policy 已覆盖真实需求 → `attempt_chain = language_chain`。
5. 子序列检查不通过 → 第一次按 `language_chain` 执行不足以完成任务 → 诊断模型假设之后追加真实最小恢复链 → `attempt_chain = language_chain + ground_truth_chain`。

只做抽象 chain 推理，不读 `clock_wise` / 几何 / 视频；它表达的是"在某个语言条件下（假设被严格遵循、无失误），若真实状态属于某条最小链，理论上会观察到什么"。如果 rollout 实际有夹爪松开 / 门回关 / 二次尝试等物理细节，实测 attempt 可能比抽象结果更长——本函数无法自动覆盖那些情况。

### 1.8 `build_attempt_partitions(language_chain, expanded_minimal_chains) -> Dict[Tuple[str, ...], List[int]]`

枚举每条候选 ground truth chain（取自 `expanded_minimal_chains`）对应的 `attempt_chain`，按 attempt 分组。返回 `Tuple[str, ...]` → 该 attempt 对应的 chain id 列表。

用途：评估某个 `language_chain` 用作探针时"信息量"——某个分组只有 1 个 chain id 就说明观察到该 attempt 能唯一识别真实状态；分组里 chain id 越多说明 attempt 含糊。

### 1.9 `infer_reasonable_prediction_chains(language_chain, ground_truth_chain=None, expanded_minimal_chains=None) -> List[List[str]]`

asker 只能根据本轮**实际表现**返回的 chain（不是枚举所有隐藏 ground-truth 状态）。规则：

1. 若提供了 `ground_truth_chain`，只针对这一个真实状态算 `observed_attempt`。
2. 若没提供，则必须传 `expanded_minimal_chains`：把每条候选当一种可能 ground truth、逐个 `infer_attempt_chain(language_chain, candidate_chain)`，得到全部可能 attempt。
3. 对每个候选 ground truth：若 `language_chain` 已覆盖它（`chain_satisfies_ground_truth=True`），实际就按 `language_chain` 跑通——`language_chain` 是合理预测。
4. 否则第一次尝试不足、需要执行真实恢复链——`ground_truth_chain` 候选本身是合理预测。
5. 若完整 `observed_attempt` 也是 `expanded_minimal_chains` 中的合法链，把它也加入合理预测集合（当前模板下通常与上面预测重复，但向前兼容更复杂的任务）。
6. 输出按出现顺序去重。

例 1：`language=["按按钮","拉门"]`、`ground_truth=["拉门"]` → `observed_attempt=["按按钮","拉门"]`。隐藏状态可能是"未锁"或"需要按按钮"，但 asker 只能看到实际执行了 `["按按钮","拉门"]`，所以返回 `["按按钮","拉门"]` 是合理的，即使它不严格等于 `ground_truth`。

例 2：只知道 `language=["拉门"]`，结合 microwave 候选全集枚举：当 ground truth 为 `["拉门"]` 时可能看到 `["拉门"]`；当 ground truth 为 `["按按钮","拉门"]` 时可能看到 `["拉门","按按钮","拉门"]`——所以合理预测集合 = `[["拉门"], ["按按钮","拉门"]]`。

### 1.10 `score_language_chain_for_inference(language_chain, expanded_minimal_chains) -> Dict[str, float]`

对单条 `language_chain` 算它当探针时的信息量指标，返回：

- `unique_ground_truth_count` / `unique_ground_truth_rate`：能被唯一识别的 ground truth 数量 / 比例。**越大越好**。
- `worst_case_candidate_count`：最坏情况下同一个 attempt 还剩多少候选 ground truth。**越小越好**。
- `expected_candidate_count`：按 ground truth 均匀先验加权后的平均候选数。**越小越好**。
- `distinct_attempt_count`：能产生多少种不同 attempt。通常**越多越好**（区分度高）。
- `mean_attempt_atomic_length` / `language_atomic_length`：次级代价（避免在信息量相同时优先选明显更长的尝试）。

实现：用 `build_attempt_partitions` 拿到分组，再求各项统计。

### 1.11 `rank_expanded_minimal_chain_ids(expanded_minimal_chains) -> List[int]` / `sort_expanded_minimal_chains_by_inference_priority(expanded_minimal_chains) -> List[List[str]]`

把 bank 里每条 chain 当探针，按 §1.10 的指标排序：

排序 key（lexicographic）：

1. 最大化 `unique_ground_truth_count`。
2. 最小化 `worst_case_candidate_count`。
3. 最小化 `expected_candidate_count`。
4. 最大化 `distinct_attempt_count`。
5. 最小化 `mean_attempt_atomic_length`。
6. 最小化 `language_atomic_length`。
7. 用原始 chain id 做 tiebreak（保证稳定排序）。

例（microwave）：`["拉门"]` 排在 `["按按钮", "拉门"]` 前面——前者锁住时观察到 `["拉门", "按按钮", "拉门"]`，能唯一区分锁住 / 未锁；后者无论锁住与否都可能只看到 `["按按钮", "拉门"]`，无法唯一判断。

`rank_*` 返回 chain id 列表；`sort_*` 返回 chain 列表本身（指向同 bank 的引用）。

---

## 2. `BaseManipulation`：chain / ground_truth 钩子

`base_manipulation.py` 中跟 chain 直接相关的方法。每个任务子类按需 override；非 Nx 任务的默认实现就够用，4 个 Nx 任务在自己的 `collect_manip_data` 中维护额外 instance state 后调用 base 的共享 helper。

### 2.1 `canonical_minimal_chain_for_state(state) -> Optional[List[str]]`

**子类必须 override**。给定 env state（`{"clock_wise": ..., ...}`），返回该任务的"模板链"（可能含 literal `Nx`）：

- microwave: cw=0 → `["拉门"]`；cw=1 → `["按按钮", "拉门"]`。
- pen: cw=0 → `["向上提起笔盖"]`；cw=1 → `["Nx旋转笔盖", "向上提起笔盖"]`（literal `Nx`）。
- ……

是 `ground_truth_chain_for_collect` 默认实现的数据源。也是 5 个非 Nx 任务的 ground_truth 直接来源。

### 2.2 `concrete_attempt_chain_for_collect(env_id, episode_state) -> Optional[Tuple[List[str], List[bool]]]`

返回 `(attempt_chain, stage_status)`。base 默认 = `(canonical_minimal_chain_for_state(state), [True] * n)`，适合 5 个非 Nx 任务（demo 一次性成功）；4 个 Nx 任务和 microwave override 给出多次失败重试的真实 attempt（细节见 [chain_concepts.md §3](chain_concepts.md)）。

### 2.3 `ground_truth_chain_for_collect(env_id, episode_state) -> Optional[List[str]]`

返回该 env state 下的"上帝视角最优"——env state 的纯函数。

- base 默认 = `canonical_minimal_chain_for_state(state)`，5 个非 Nx 任务直接用。
- 4 个 Nx 任务自己 override，调用 §2.4 的 helper 把 `_<task>_intrinsic_n[env_id]`（在 `collect_manip_data` 中观测得到的 env 物理 N_min）填进具体 N。

### 2.4 `ground_truth_chain_from_intrinsic_n(env_id, state, n_min_attr, rotate_op, lift_op, success_hint) -> Optional[List[str]]`（**新增**）

4 个 Nx 任务 `ground_truth_chain_for_collect` override 的共享 body。读 `getattr(self, n_min_attr)[env_id]`：

| 取值 | 返回 |
|---|---|
| `0` | `[lift_op]` |
| `> 0` | `[f"{N_min}x{rotate_op}", lift_op]` |
| 缺失（`getattr` 未设、env_id 越界、值是 `None`） | warn-banner + fallback 到 `canonical_minimal_chain_for_state(state)` |

`success_hint` 字符串拼到 fallback 警告里，方便操作者识别哪个任务的 demo 没正确翻转 `open_bottle_stage`（例如 pen 传 `"opened the pen cap"`，cm 传 `"pulled the coffee-machine handle"`）。fallback 路径**不应**在正常采集中触发——成功 trajectory 必然让 `open_bottle_stage` 翻成 True、N_min 不会是 None。

各任务 override 现在缩成 5–8 行：

```python
# open_pen.py
def ground_truth_chain_for_collect(self, env_id, state):
    return self.ground_truth_chain_from_intrinsic_n(
        env_id=env_id, state=state,
        n_min_attr="_pen_intrinsic_n",
        rotate_op="旋转笔盖", lift_op="向上提起笔盖",
        success_hint="opened the pen cap",
    )
```

`_<task>_intrinsic_n` 的填充逻辑由各 `collect_manip_data` 自己负责（见 [chain_concepts.md §3.2](chain_concepts.md)）。

### 2.5 `match_command_chains(attempt_chain, stage_status, expanded_minimal_chains) -> Tuple[List[List[str]], List[int]]`

按 schema rule 算 `command_chains` / `command_chain_ids`：

1. 找 `stage_status` 中**第一个** `False` 的位置 `k`；无失败则 `k = len-1`。
2. `prefix = attempt_chain[:k+1]`。
3. 在 `expanded_minimal_chains` 中筛选所有以 `prefix` 起始的 chain；对应下标即 `command_chain_ids`。
4. 若结果为空，抛 `RuntimeError`——caller 一般会捕获并 fallback。

注意 §1.6 的 `extract_minimal_chain_from_attempt` 是看**最后一个** `False`，与本函数看**第一个** `False` 的方向相反——对应"minimal_chain 是丢弃失败后聚合得到"vs"command_chains 是按命令一致性锚定到首次失败前缀"两种语义。

### 2.6 `_build_full_minimal_chain_bank(observed_chains, trajectory_records=None) -> List[List[str]]`（**新增**）

把"实际出现过的 minimal_chain 集合" 强行展开成 `1..N_max` 全集，让 §2.5 的 prefix 匹配对中间 N 也总能命中：

1. 扫 `observed_chains`：
   - 长度 == 2、第一段匹配 `^(\d+)x(.+)$` → 取出 `N`、`op_root`、`lift`，按 `(op_root, lift)` 分组求 max N。
   - 其他 chain → 原样保留（按观察顺序去重）。
2. 若给了 `trajectory_records`，再扫每条 `attempt_chain` 的相邻"`Nx{op}` + `lift`"对，一并并入 `(op_root, lift)` 分组的 max N（解决 prefix 上的 N 可能比任何 minimal_chain 都大的情况）。
3. 输出顺序：先 non-Nx chain（保留观察顺序），再每个 `(op_root, lift)` 按 `1x..N_max{op_root}` + `lift` 顺序展开。

这是 `expanded_minimal_chains`（最终落到 `language_expanded.json` 的 chain id 索引表）的真正构造函数。子类可 override 来改 `N_max` 上界（例如基于 env 物理参数推一个解析上界）；目前所有任务用默认实现。

### 2.7 `extract_minimal_chain_from_attempt(...)`（来自 §1.6）

`base_manipulation.py::collect_episode_end` 直接调用 §1.6 的同名函数算每条 trajectory 的 `minimal_chain`。详见 §1.6。

---

## 3. `BaseManipulation`：日志辅助

### 3.1 `get_logger() -> Optional[Logger]`（**新增**）

包装 `getattr(self, "logger", None)`。子类不用每次都防御性 check，写 `logger = self.get_logger(); if logger is not None: logger.warning(...)` 即可。

### 3.2 `warn_banner(message: str)`（**新增**）

把 `message` 用 100 个 `!` 包夹打印到 stdout（`flush=True`），同时通过 `get_logger().warning(message)` 写日志（如果 logger 存在）。

适用场景：**理论上不该走到、但又必须返回个值不能 raise** 的代码路径——例如 §2.4 的 fallback。设计上故意吵闹，让操作者交互运行 collect 时立刻看到异常情况。

```text
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
[GROUND_TRUTH FALLBACK] OpenPenManipulation: env_id=3 (_pen_intrinsic_n[3]=None ...);
falling back to canonical_minimal_chain_for_state (literal Nx). Verify the demo
actually opened the pen cap.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

---

## 4. 关联文档

- 概念性介绍 / 各 chain 字段语义：[`chain_concepts.md`](chain_concepts.md)。
- 采集 schema 规约：[`data_collection.md`](data_collection.md)。
- 推理 / 评估流程总览：[`../inference.md`](../inference.md)（其 §8.3 chain 推理优先级讨论引用本文档的 §1.6–§1.11）。
- 工具函数手写示例脚本：[`tests/show_language_chain_reasoning_examples.py`](../../tests/show_language_chain_reasoning_examples.py)。
