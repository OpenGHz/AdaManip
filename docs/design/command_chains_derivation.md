# `command_chains` 推导全流程

本文档把 `command_chains` / `command_chain_ids` 从 **demo 执行** 到 **落盘 `trajectory_language.jsonl`** 再到 **dataloader 消费** 的完整推导链路串起来，重点讲两件别处没集中说清的事：

1. **端到端怎么算出来**（prefix 规则 + bank 前缀匹配）；
2. **为什么要算两次**（`collect_episode_end` 临时算 vs `collect_finalize` 权威重算）。

> 分工：chain 字段的**语义概览**见 [`chain_concepts.md`](chain_concepts.md) §4；`match_command_chains` / `_build_full_minimal_chain_bank` 的**单函数签名**见 [`chain_utils_reference.md`](chain_utils_reference.md) §2.5 / §2.6。本文档是把它们 + 计算时机串成 pipeline 的那一份。

---

## 1. 一句话与语义

`command_chains` = **把 demo "第一次下达的完整命令"当查询键，在 bank（`expanded_minimal_chains`）里捞出所有以它为前缀的候选完整链**（`List[List[str]]`）。`command_chain_ids` 是这些链在 bank 中的下标。

它的语义是"**命令一致性优先**"：锚定 demo 第一拍承诺要做的事，而不是 demo 最终怎么成功、也不是 env 的上帝视角真值。训练时 dataloader 从这个候选集里采一条当语言条件。

> 与另外两条 chain 的对照（详见 [`chain_concepts.md`](chain_concepts.md)）：
> - `minimal_chain`：demo 真实成功路径聚合后的最简形式（看**最后一个**失败之后的累计）。
> - `ground_truth_chain`：env 物理的上帝视角最优（与 demo 无关）。
> - `command_chains`：demo **首次失败前**的命令前缀能匹到的候选集（看**第一个**失败）。

## 2. 核心算法：prefix + bank 前缀匹配

实现：`base_manipulation.py::match_command_chains(attempt_chain, stage_status, expanded_minimal_chains)`。三步：

1. **找第一次失败** `first_fail` = 首个 `stage_status == False` 的下标；若全 `True` 则 `end_idx = len-1`。
2. **切 prefix** = `attempt_chain[:end_idx + 1]` —— 切片**包含**那次失败的 stage 本身。
3. **bank 上前缀匹配**：遍历 `expanded_minimal_chains`，凡满足 `len(chain) >= len(prefix)` 且 `chain[:len(prefix)] == prefix` 的链全部收进 `command_chains`，下标进 `command_chain_ids`。空结果 → 抛 `RuntimeError`（caller 捕获并 fallback，见 §5）。

### 2.1 为什么 prefix 含失败段

demo 跑 `["9x旋转笔盖", "向上提起笔盖(fail)", ...]` 时，它第一拍"想做的事"就是"转 9 次再抬"——那次失败的 lift 正是这条命令的一部分。所以 `prefix = ["9x旋转笔盖", "向上提起笔盖"]`，命中 bank 里 N=9 的那条链。若把失败段排除在 prefix 外，就只剩 `["9x旋转笔盖"]`，会同时命中 N=9 的所有 `9x..` 链且语义模糊。**含失败段 = 锚定"完整的一条命令"**。

> 这与 `minimal_chain` 的取向相反：`minimal_chain` 看**最后一个**失败（丢弃失败后聚合），`command_chains` 看**第一个**失败（命令一致性锚定）。两个方向对应两种语义，见 `chain_utils_reference.md` §2.5 末尾。

### 2.2 通常命中 1 条

对 pen / bottle / pc / cm 这类 Nx 任务，bank 里每个 N 的链唯一，且 lift-only 链开头是 `向上提起` 而 Nx 链开头是 `ix旋转`，互不前缀包含。所以 prefix 在实践中**恰好命中 1 条**，`command_chains` 是单元素列表。`List[List[str]]` 只是为兼容"一个前缀对应多条候选"的通用形态而保留。

## 3. 前缀为何总能命中：bank 强行展开

prefix 里的 N（第一次失败点，如 9）往往**不是任何 env 的最终 `minimal_chain`**（大家都转到 15/17 才成功）。若 bank 只装"实际出现过的 minimal_chain"，prefix=9 就匹不到、要走 fallback。为消灭 fallback，`collect_finalize` 把 bank 强行展开到 `1..N_max` 全集。

实现：`base_manipulation.py::_build_full_minimal_chain_bank(observed_chains, trajectory_records)`：

- 非 Nx 链（`["向上提起笔盖"]`、`["按按钮", "拉门"]`）原样保留、去重保序。
- Nx 链按 `(op_root, lift)` 分组，输出 `1x{op} .. N_max x{op}` + `lift`。

**`N_max` 有两个来源**（关键）：

1. 各条 `minimal_chain` 里出现的最大 N；
2. **额外扫每条 `attempt_chain` 的相邻 "`Nx{op}` 紧跟一个非 Nx(lift)" 对**，把那个 N 也并入 max。

第 2 个来源专为 `command_chains` 服务：prefix 的 N（首次失败点）可能比任何 `minimal_chain` 的 N 都不同甚至更大，不扫 attempt 就会漏掉它、导致前缀匹配失败。

> 因此"schema 规定 `command_chains` 必须非空"这个硬约束**结构上自然成立**，不依赖任何 fallback 路径（§5 的 fallback 仅为防御性兜底）。

## 4. 计算时机：算两次，以第二次为准

`command_chains` 在采集流程里被算两次，因为第一次算的时候 bank 还没定稿。

| | 第一次（临时） | 第二次（权威） |
|---|---|---|
| 位置 | `base_manipulation.py::collect_episode_end` | `base_manipulation.py::collect_finalize` |
| 触发 | 每条成功 trajectory 落盘时 | 所有 episode 跑完后一次性 |
| 此刻 bank 状态 | 增量 append、**未展开**，prefix 的中间 N 多半不在 | 已过滤 abstract + 强行展开到 `1..N_max`，定稿 |
| 结果 | 临时、常落 fallback `[minimal_chain]` | 权威，写进 `trajectory_language.jsonl` |

`collect_finalize` 的顺序：

1. 过滤掉模板残留的含字面 `Nx` 的 abstract 链 → `expanded_actual_minimal_chains`；
2. `_build_full_minimal_chain_bank(...)` 强行展开 → 最终 `expanded_minimal_chains`；
3. **对每条 record 重新跑 `match_command_chains`**，覆写 `command_chains` / `command_chain_ids`（同时重算 `minimal_chain_id`）；
4. 把两个 bank + 所有 record 写进 `language_expanded.json` / `trajectory_language.jsonl`。

> 之所以拖到 finalize：`collect_episode_end` 触发时 bank 仍在动态扩展（每 epoch 末才追加新观察到的 chain），早期算出的 chain id 在最终 bank 里可能失效。finalize 末尾统一重算一次，保证所有 trajectory 看到同一个最终 bank、id 对得上 embedding 表。

## 5. fallback 路径

- **`collect_episode_end`（第一次）**：bank 未展开，`match_command_chains` 抛 `RuntimeError` 是常态 → `except` 落 `command_chains = [minimal_chain]`、`command_chain_ids = [minimal_chain_id]`。env 自己的最优链永远是一个合法的命令解释，所以这个兜底语义安全。
- **`collect_finalize`（第二次）**：全集 bank 已覆盖所有 prefix，**理论上不该再抛**。若真抛了，说明 bank 展开逻辑有 bug → `logger.warning` 吵闹报警 + 同样 fallback 到 `[minimal_chain]`，避免整批采集丢失。

## 6. 三个常见 case（pen，bank 已展开 1..N_max）

| 场景 | attempt_chain 起始 | first_fail | prefix | `command_chain_ids` |
|---|---|---|---|---|
| cw=0，首拍直接抬升成功 | `向上提起笔盖` | 无（全 True） | `["向上提起笔盖"]` | `[0]`（只命中 lift-only 链） |
| cw=1，首拍误抬失败，后续再转直到成功 | `向上提起笔盖` | 0 | `["向上提起笔盖"]` | `[0]`（**与本 env 真实 cw=1 不符**，见下） |
| cw=1，先转 K 次再抬失败、继续重试 | `Kx旋转笔盖` | 1 | `["Kx旋转笔盖", "向上提起笔盖"]` | `[K]` |

> case 2 是最容易踩的偏置：`command_chains` 反映 **demo 第一拍的意图**（误以为 cw=0、直接抬），不是 env 真值。对 cw 一致性敏感的训练应改走 `minimal_chain_id` 或 `ground_truth_chain` 绕过它。

## 7. 端到端例子（接 `chain_concepts.md` §7 的 pen trajectory）

某条 cw=1 的 pen trajectory，demo 真实动作流 `r…r z r z r z r z r…r z`：

```jsonc
"attempt_chain": ["9x旋转笔盖","向上提起笔盖","1x旋转笔盖","向上提起笔盖",
                  "1x旋转笔盖","向上提起笔盖","1x旋转笔盖","向上提起笔盖",
                  "5x旋转笔盖","向上提起笔盖"],
"stage_status":  [true,false, true,false, true,false, true,false, true,true]
```

- `first_fail = 1`（第一次抬升失败）→ `prefix = attempt_chain[:2] = ["9x旋转笔盖", "向上提起笔盖"]`。
- bank 全集里下标 9 即此 chain（`N_max` 因为扫了 attempt 的 `9x旋转 + 向上提起` 对，至少覆盖到 9）→ `command_chains = [["9x旋转笔盖", "向上提起笔盖"]]`、`command_chain_ids = [9]`。
- 对照同一条的另外两个 chain：`minimal_chain = ["17x旋转笔盖", "向上提起笔盖"]`（丢 4 次失败 lift + 合并 4 段 rotate，K=17）；`ground_truth_chain = ["15x旋转笔盖", "向上提起笔盖"]`（env 物理 N_min=15）。三者各看一个侧面。
- `["9x..."]` 不在 `expanded_actual_minimal_chains`（没有 trajectory 以 9x 收尾），但强行展开后它在 `expanded_minimal_chains[9]` → `command_chains` 永远非空、不走 fallback。

## 8. 下游消费

`dataset/dataset.py`（ManipDataset 启用 language conditioning 时）：

- 读每条 trajectory 的 `command_chain_ids`，**强制非空**（空则 `ValueError`），并校验下标落在 `expanded_minimal_chains` 范围内（越界 `IndexError`）。
- 把 `(source_idx, chain_ids)` 存入 `_episode_lang_index`；训练 `__getitem__` 时从这组 id 里采一条，用 `language_embedding_dict.json["expanded_minimal_chains"][cid]` 的向量当语言条件。

§3 的"bank 强行展开 + prefix 总命中"这条链路，最终目的就是保证这里的 `command_chain_ids` **永远非空、永远对得上 embedding 表下标**。

## 9. 参考代码位置

| 环节 | 位置 |
|---|---|
| prefix + 前缀匹配 | `manipulation/base_manipulation.py::match_command_chains`（签名见 `chain_utils_reference.md` §2.5） |
| bank 强行展开 + N_max 两来源 | `manipulation/base_manipulation.py::_build_full_minimal_chain_bank`（签名见 `chain_utils_reference.md` §2.6） |
| 第一次（临时）计算 + fallback | `manipulation/base_manipulation.py::collect_episode_end` |
| 第二次（权威）重算 + 落盘 | `manipulation/base_manipulation.py::collect_finalize` → `save_language_sidecars` |
| 下游消费 | `dataset/dataset.py`（`command_chain_ids` 校验 + 采样） |
| 概念语义概览 | [`chain_concepts.md`](chain_concepts.md) §4 |
