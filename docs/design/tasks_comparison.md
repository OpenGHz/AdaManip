# 任务相同点 / 差异点对照

本文档把现有 9 个任务（`microwave` / `safe` / `door` / `window` / `lamp` / `bottle` / `pen` / `pressure_cooker` / `coffee_maker`）按几个关键维度做横向对照，方便快速看清"哪些是共性、哪些是任务自定义"。每个任务的具体语义见 [`tasks.md`](tasks.md)，chain 字段含义见 [`chain_concepts.md`](chain_concepts.md)，各 chain 工具函数实现见 [`chain_utils_reference.md`](chain_utils_reference.md)。

---

## 1. 整体分类

### 1.1 数据采集结构：unified vs split

| 类别 | 任务 | 采集脚本 |
|---|---|---|
| **unified**（只有 `manip` 一种采集） | `microwave`、`safe` | `collect_microwave_manip.sh` / `collect_safe_manip.sh` |
| **split**（`manip` + `grasp` 分两步采集，推理时由两个 net 串联） | `door`、`window`、`lamp`、`bottle`、`pen`、`pressure_cooker`、`coffee_maker` | `collect_<task>_manip.sh` + `collect_<task>_grasp.sh` |

split 任务的 `cfg.task.grasp` / `cfg.model.grasp` 区分采集 / 推理两侧的开关；unified 任务上述两个 cfg 字段都为 `False`。

### 1.2 minimal_chain 是否含 `Nx`（Nx vs non-Nx）

| 类别 | 任务 | 旋转 op | lift / final op | template `minimal_chains` 形态 |
|---|---|---|---|---|
| **non-Nx** | `microwave` | – | `拉门`（兼按按钮）| 单 stage 或两 stage 固定 |
| **non-Nx** | `safe` | `顺时针旋转旋钮` / `逆时针旋转旋钮` | `拉门` | `[lift]` 或 `[rotate, lift]` |
| **non-Nx** | `lamp` | `顺时针旋转开关` / `逆时针旋转开关` | `推开关` | 单 stage |
| **non-Nx** | `door_one_go` / `window_one_go` | `顺时针旋转把手` / `逆时针旋转把手` | `拉开门` / `拉开窗户` | `[rotate, lift]` 两 stage 固定（one_go 单步旋转跨阈值） |
| **Nx** | `door` / `window` | `顺时针旋转把手` / `逆时针旋转把手` | `拉开门` / `拉开窗户` | `[lift]` 或 `[Nx rotate, lift]`，N 由 env 物理决定 |
| **Nx** | `bottle` / `pen` / `pressure_cooker` / `coffee_maker` | `旋转<part>` | `向上提起<part>` 或 `拉动手柄` | `[lift]` 或 `[Nx rotate, lift]`，N 由 env 物理决定 |

`Nx` 任务的特殊之处：

- 同一 chain 在 trajectory 之间可能有不同的具体 N（例如 `12x旋转笔盖` vs `19x旋转笔盖`），bank 必须强行展开成 `1..N_max` 全集才能保证 prefix 匹配总命中（详见 [chain_concepts.md §5](chain_concepts.md)）。
- ground_truth_chain 的具体 N 不能从 cw 一个 bit 得到；用 `open_bottle_stage` 翻转点观测（详见 [chain_concepts.md §3.1](chain_concepts.md)）。

---

## 2. `clock_wise` 字段的含义

所有任务的 env state 都包含 `clock_wise`（per-env，多取 0 / 1，少数取 2 / 3）。但**这一 bit 表示什么**因任务而异：

| 任务 | cw=0 / cw=1 / 其他取值 | 是否真区分"需不需要 rotate" |
|---|---|---|
| `microwave` | 0=未锁；1=锁着（需要按按钮） | ✓ 真区分（cw=0 时 `[拉门]` 就够了） |
| `safe` | 0=未锁；1=顺时针解锁；2=逆时针解锁（特殊取值，多 cw 状态） | ✓ 真区分 |
| `door` / `window` | 1=顺时针；其它=逆时针 | ✗ 只决定**旋转方向**，两边都要 rotate |
| `lamp` | 1=推；3=顺时针；2=逆时针；其它=N/A | ✓ 决定操作类型（推 vs 转） |
| `bottle` / `pen` | 1=顺时针；0=逆时针 | ✗ 只决定方向；都要 rotate |
| `pressure_cooker` | 1=需 rotate；0=直接 lift | ✗（默认 cfg `clockwise=0.0` 即恒定 cw=0；env 内部仍是锁着） |
| `coffee_maker` | 1=需 rotate；0=直接 pull | ✗（默认 cfg `clockwise=1.0` 即恒定 cw=1） |

**坑**：cfg 里的 `cfg.env.clockwise` 是**这个 episode 抽到 cw=1 的概率**（伯努利系数），不是直接的 cw 值。比如 `clockwise=0.5` 就一半 cw=0、一半 cw=1；`clockwise=1.0` 则全部 cw=1。

---

## 3. 每个 trajectory 的 `clock_wise` 怎么决定

实现都在 `envs/open_<task>.py::init_obj_dof_state()` / `_partial_reset()` 中。两条路径：

1. **`clock_wise_override`**（adaptive eval state machine 用的）：episode 1 之后把 cw 写死，复用 episode 1 sample 出来的值，避免每次 reset 都重新抽。
2. **`np.random.rand() < cfg.env.clockwise`**（默认采集路径）：每个 env 独立 Bernoulli sample。

`reset_kwargs_initial()` 钩子控制默认 `clock_same`：`microwave`、`safe`、`door` 显式 override 成 `clock_same=False`（每个 env 独立 sample），其余任务用 base 默认（每个 episode 全部 env 共用同一个 cw，参考 [`chain_concepts.md`](chain_concepts.md) 实测时见到的 `clock_same=True` 行为）。

---

## 4. 任务子类 override 的钩子矩阵

✓ 表示该任务 override 了对应方法。（完整方法签名见 [`chain_utils_reference.md`](chain_utils_reference.md) §2 和 [`base_manipulation.py`](../../manipulation/base_manipulation.py)。）

| 钩子 | microwave | safe | door | window | lamp | bottle | pen | pc | cm |
|---|---|---|---|---|---|---|---|---|---|
| `language_template_task_name` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `task_success_for_env` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `capture_per_env_episode_state` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `apply_frozen_states_to_reset_kwargs` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `canonical_minimal_chain_for_state` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `per_env_extra_log_fields` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `reset_kwargs_initial` | ✓ | ✓ | ✓ | – | – | – | – | – | – |
| `concrete_attempt_chain_for_collect` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `ground_truth_chain_for_collect` | – | – | ✓ | ✓ | – | ✓ | ✓ | ✓ | ✓ |
| `dataset_dir_suffix` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `collect_grasp_data` | – | – | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `collect_manip_data` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `diffusion_eval_grasp` | – | – | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

观察：

- 7 个钩子是**所有 9 个任务都 override** 的"任务身份"基础（task_name、cw 状态记录与还原、success 判定、canonical chain、log 字段）。
- `concrete_attempt_chain_for_collect`：**所有 9 个任务**都 override，因为每个任务的 adaptive demo 都可能产生重试/失败-切换片段（microwave 多次按按钮拉门；safe 三向初始动作 + 失败 pull 切 rotate；door / window / lamp 试错方向；4 个 Nx 任务的转-提循环）。
- `ground_truth_chain_for_collect`：6 个 Nx 任务 override（door / window / bottle / pen / pressure_cooker / coffee_maker），需要从 env 物理记录的 `intrinsic_n` 填具体 N 才能给出 ground-truth chain；其它 3 个 non-Nx 任务（microwave / safe / lamp）用 base 默认 `canonical_minimal_chain_for_state` 即可。这些 override 在 grasp-data 流程下会通过 `getattr(self, '_<task>_intrinsic_n', None) is None` 检测到 attribute 未初始化、静默回退到 canonical chain，避免 grasp 期间刷屏 warning。
- `reset_kwargs_initial`：3 个 unified / 简单任务（microwave / safe / door）显式声明 `clock_same=False`；其余依赖 env 默认 `clock_same=True`。
- split 任务都 override `collect_grasp_data` + `diffusion_eval_grasp`；unified 任务不需要。

---

## 5. demo 策略形态

| 任务 | 形态 | 关键随机化 |
|---|---|---|
| `microwave` | 确定性脚本 + 1 bit 随机 | `start_with_pull` ~ Bernoulli(0.5)：决定先试 `拉门` 还是先试 `按按钮` |
| `safe` | 类似 microwave，多分支判断（先 pull / 先 cw rotate / 先 ccw rotate） | `start_with` ~ {pull, cw_rotate, ccw_rotate} |
| `door` / `window` | 简单脚本：抓把手 → 朝固定方向 rotate → 拉开 | （非 adaptive 时确定性） |
| `lamp` | 单步操作，按 cw 选 push / cw rotate / ccw rotate | （非 adaptive 时确定性） |
| `bottle` / `pen` / `pc` / `cm` | per-env state machine，逐拍 ada_policy 决定 `r` / `o` / `z` | `ada_policy` 在过 `try_range` 之后以 `prob = 11/20` 切到 `z`，否则继续 rotate |

4 个 Nx 任务的 ada_policy 共用思路（详见 [`data_collection.md §10.3`](data_collection.md)）：在 `|dof| < try_range` 时强制 rotate；过了 try_range 之后按概率切 lift；lift 失败就再 rotate；这是 attempt_chain 出现"多次 rotate-lift 循环"的根源。

---

## 6. 关键数值参数对照

数据来自 `cfg/<task>/collect_<task>_manip.yaml`（默认采集 cfg）和各 `manipulation/open_<task>.py` 的 `collect_manip_data`：

| 任务 | numEnvs | num_episode | horizon | clockwise | limit_random | max_step (adaptive) | max_step (succ) | success 阈值 | env try_range |
|---|---|---|---|---|---|---|---|---|---|
| microwave | 10 | 25 | 30 | 0.5 | – | – | – | `&#124;one_dof&#124; > π/7` | – |
| safe | – | – | – | – | – | – | – | `&#124;one_dof&#124; > π/7` | – |
| door | 8 | 20 | 35 | 0.5 | 0.5 | 30 | 25 | `&#124;one_dof&#124; > π/7` | – |
| window | 8 | 20 | 35 | 0.5 | 0.5 | 25 | 20 | `&#124;one_dof&#124; > π/6` | – |
| lamp | 7 | 20 | 30 | 0.5 | 0.5 | 15 | 10 | 任务依赖（按 cw 选 one_dof 或 two_dof） | – |
| bottle | 7 | 20 | 50 | 0.5 | 0.5 | 25 | 20 | `&#124;one_dof&#124; > 0.04` | 0.99875 |
| pen | 10 | 1 (smoke) | 35 | 0.5 | 0.5 | 30 | 25 | `&#124;one_dof&#124; > 0.025` / 0.04 | 0.99875 |
| pressure_cooker | 6 | 20 | 30 | 0.0 | 0.5 | 25 | 20 | `&#124;one_dof&#124; > 0.025` | 0.35 |
| coffee_maker | 7 | 20 | 30 | 1.0 | 0.5 | 20 | 15 | `&#124;one_dof&#124; > 0.025` | 0.34 |

> `try_range` 是 4 个 Nx 任务的 ada_policy 在 env 端的"够松了"阈值，写死在 `envs/open_<task>.py::__init__`，不进 cfg。`limit_random` 是 cap 旋转关节随机限位的尺度（决定真实 N_min 范围），见 [`data_collection.md §10`](data_collection.md)。
> 安全阈值（success）逐任务不同；`pen` 里 `collect_manip_data` 用 0.04，`task_success_for_env` 用 0.025（前者是采集时判断"这条 trajectory 算不算成功"，后者是 eval 时判断 trajectory 是否完成；阈值差异是为了让采集只保留充分张开的轨迹）。

---

## 7. attempt_chain 出现的形态

| 形态 | 出现条件 | 任务 |
|---|---|---|
| 单 stage 一次成功 | demo 一拍把任务做完 | door / window / lamp / safe（cw=0 unsafe）/ pen-bottle-pc-cm 的 cw=0 demo 直接抬升的 trajectory |
| 两 stage 一次成功 | demo 按命令两步连贯 | microwave cw=1 一次完成、door / window cw=0 与 cw=1 时分别 cw / ccw 旋转 |
| 三 stage 含一次失败重试 | microwave 先误拉门后按按钮 | microwave cw=1 / start_with_pull=True |
| 多段 rotate↔lift 循环 | demo `ada_policy` 反复 rotate→fail-lift→rotate→succ-lift | bottle / pen / pc / cm 的所有 cw=1（且通常的 cw=0）trajectory |

---

## 8. eval 阶段的特殊性

| 任务 | adaptive eval 是否启用语言条件分支 | 备注 |
|---|---|---|
| 都支持 `cfg.task.adaptive_language.enable` 的开关 | 默认 false，开启后用 `AdaptiveLanguageState` state machine 选 chain id | 详细见 [`../inference.md §8`](../inference.md) |

eval 时 `expanded_minimal_chains` 来自数据预处理产出的 `language_embedding_dict.json`（`docs/design/data_preprocess.md`）。所有任务用同一套 `BaseManipulation.diffusion_evaluate` 流程；任务差异通过 §4 的钩子矩阵插入（success 判定、canonical chain、cw 还原等）。

---

## 9. 一句话快速记忆

| 任务 | 一句话 |
|---|---|
| microwave | 拉门；按了门锁的需要先按按钮再拉。**non-Nx**，cw 真区分；demo 单次成功或 1 次重试。 |
| safe | 拉门；锁住的需要先按方向（cw 或 ccw）旋转旋钮再拉。**non-Nx**，cw 真区分；demo 多分支。 |
| door | 抓把手 → 顺时针 / 逆时针旋转 → 拉开门。**non-Nx**，cw 决定方向；split。 |
| window | 同 door 但操作的是窗户。**non-Nx**，split。 |
| lamp | 一步动作：推开关 / 顺时针旋转 / 逆时针旋转，三选一。**non-Nx**，cw 决定操作类型；split。 |
| bottle / pen | 抓盖子 → 转 N 次 → 抬起。**Nx**，N 由 env 物理决定；split。 |
| pressure_cooker | 抓把手 → 转 N 次 → 抬起。**Nx**，cfg `clockwise=0.0` 默认；split。 |
| coffee_maker | 抓手柄 → 转 N 次 → 拉下。**Nx**，cfg `clockwise=1.0` 默认；split。 |

---

## 10. 关联文档

- 每个任务的语义、操作集合、最优链：[`tasks.md`](tasks.md)。
- chain 字段含义对照：[`chain_concepts.md`](chain_concepts.md)。
- chain 工具函数实现细节：[`chain_utils_reference.md`](chain_utils_reference.md)。
- 采集 schema 与 N 范围由什么决定：[`data_collection.md`](data_collection.md)。
- 训练数据加载 + 多文件合并：[`training_data_loading.md`](training_data_loading.md)。
- 推理 / adaptive eval：[`../inference.md`](../inference.md)。
- env 端的随机化（cw / dof 限位）：[`../randomization.md`](../randomization.md)。
- 加新任务的步骤：[`../../../docs/new_task_guide.md`](../../../../docs/new_task_guide.md)。
