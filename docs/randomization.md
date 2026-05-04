# Randomization

## open_microwave

数据采集时有两层独立的随机化，共同决定每条轨迹的策略。

### 1. 机构随机化（`clockwise`）

**配置项：** `env.clockwise`（默认 0.5）

每次 reset 时，每个 env 独立抽样，以 `clockwise` 为概率将该 env 的门设为"需要按钮解锁"：

```yaml
env:
  clockwise: 0.5   # 50% 的 env 需要先按按钮才能开门
```

- `clock_wise=1`：门的 DOF 上限初始为 0，直接拉门无效，必须先按按钮才能解锁（DOF 上限扩展至 1.578）
- `clock_wise=0`：门可以直接拉开，无需按按钮

### 2. 策略随机化（`start_with_pull`）

**代码位置：** `manipulation/open_microwave.py` adaptive policy，`np.random.rand() < 0.5`

每个 episode 开始时随机决定机械臂的初始尝试动作，**所有 env 共享同一个决策**（episode 级别）：

- `start_with_pull=True`（50%）：先尝试直接拉门
- `start_with_pull=False`（50%）：直接执行"按按钮→拉门"

### 组合结果

策略分支由 **env 0 的 `clock_wise`** 决定（`manipulation/open_microwave.py:615`：`if start_with_pull and self.env.clock_wise[0] == 0`），所有 env 共享同一条动作序列。当 `start_with_pull=True` 时，无论 `clock_wise` 如何都会先执行一段"拉门尝试"动作。

| `start_with_pull` | `clock_wise[0]` | 实际轨迹 | 概率 |
|---|---|---|---|
| True | 0 | 拉门 → 开门成功 | 25% |
| True | 1 | **拉门 → 拉不动 → 按按钮 → 再拉门**（失败恢复，adaptive 的核心轨迹） | 25% |
| False | 0 | 按按钮 → 拉门 | 25% |
| False | 1 | 按按钮 → 拉门 | 25% |

策略是 episode 级别的（全 env 统一），门的机构 `clock_wise` 是 env 级别独立采样的；但策略分支只看 env 0 的 `clock_wise`。当 `start_with_pull=True` 且 `clock_wise[0]=0` 时，那些自身 `clock_wise=1` 的 env 会因门被锁定开门角度不足而被 `manipulation/open_microwave.py:702` 过滤丢弃。

**注意：** 单一路径仅占 25% 概率，少量 episode 集中在某条路径是正常现象（3 个 episode 全为按钮路径概率约 42%）。增加 episode 数后四种轨迹都会出现，"先拉失败再按按钮"是 adaptive 策略的核心采集目标。

### 调整建议

| 目标 | 配置 |
|---|---|
| 采集纯"直接开门"数据 | `clockwise: 0.0` |
| 采集纯"按按钮开门"数据 | `clockwise: 1.0` |
| 两种策略各 50% | `clockwise: 0.0`（消除机构随机化干扰） |

## 推理阶段随机化

推理脚本 [scripts/microwave/eval_microwave_model.sh](../scripts/microwave/eval_microwave_model.sh)
通过 `ADA_MANIP_SEED` 控制 seed，默认是 `0`：

```bash
ADA_MANIP_SEED=42 \
ADA_MANIP_CFG_ENV=cfg/microwave/exp_ground_truth_prompt_eps5.yaml \
sh third_party/ada_manip/scripts/microwave/eval_microwave_model.sh
```

脚本会把同一个 seed 同时传给 rpyc server 和 client：

```text
--seed="$SEED"
```

`run.py` 中的 `set_seed(args.seed)` 会设置 Python `random`、NumPy、PyTorch CPU/CUDA
随机种子，并把最终 seed 写入输出目录的 `eval_config.yaml`。因此判断某次推理 seed 是否生效，
最直接的方法是查看对应 run 目录里的：

```yaml
seed: 42
```

### 1. 机构状态随机化（`clock_wise`）

推理中仍会随机采样微波炉是否需要按钮解锁。配置项仍然是：

```yaml
env:
  clockwise: 0.5
```

在 `OpenMicroWaveManipulation.diffusion_evaluate()` 中，非自适应模式的每个 episode，以及自适应模式的
第 1 个 episode，都会调用：

```python
self.env.reset(clock_same=False)
```

因此 `envs/open_microwave.py` 会对每个 env 独立采样：

```python
np.random.rand(self.env_num) < self.cfg["env"]["clockwise"]
```

也就是说：

- `env.clockwise=0.5`：每个 env 独立以 50% 概率成为 `clock_wise=1`
- `env.clockwise=0.0`：所有 env 都是 `clock_wise=0`
- `env.clockwise=1.0`：所有 env 都是 `clock_wise=1`

自适应模式下，第 1 个 episode 结束后会冻结每个 env 的 `clock_wise`：

```python
s.frozen_clock_wise = float(cw[env_id])
```

后续 episode reset 时使用：

```python
self.env.reset(clock_wise_override=override)
```

所以后续 episode 不再重新采样机构状态，而是复用第 1 个 episode 的 env 状态。

### 2. 非自适应推理的 prompt 随机化

当 `task.adaptive_language.enable=false` 时，每个 episode 会随机采样一个语言 embedding id，
并广播给所有 env：

```python
sampled_idx = int(np.random.randint(0, bank_size))
```

因此 Random prompt baseline 中：

- 每个 episode 的 prompt 是随机的；
- 同一 episode 内所有 env 使用同一个 prompt；
- 该随机性受 `ADA_MANIP_SEED` 影响。

### 3. 自适应推理的 prompt 尝试顺序

当 `task.adaptive_language.enable=true` 时，每个 env 都有自己的 `AdaptiveLanguageState`。
如果 env 还没有锁定 prompt，会调用：

```python
cid = s.pick_next(adaptive_rng, priority_ids=adaptive_chain_priority_ids)
```

其中 `adaptive_rng = random.Random(seed)`。不过当前代码会传入
`adaptive_chain_priority_ids`，此时 `pick_next()` 会按优先级返回第一个还没尝试过的 chain id，
不会调用 `rng.choice()`。也就是说，在当前自适应推理路径中，prompt 尝试顺序主要由
`rank_expanded_minimal_chain_ids()` 的确定性排序决定；只有没有提供优先级列表时，才会退化为
受 `ADA_MANIP_SEED` 控制的随机选择。

当前 microwave 的 expanded minimal chains 只有两条：

```text
0: 拉门
1: 按按钮 -> 拉门
```

推理优先级由 `rank_expanded_minimal_chain_ids()` 计算。由于 `拉门` 更能区分真实机构状态，
通常第 1 个 episode 会优先尝试 chain id `0`。如果优先级排序已经给出明确首选项，
改变 seed 不会改变第 1 个 episode 的 `language_chain_id`；seed 主要影响机构状态采样，
以及未来如果关闭优先级排序后的随机 fallback。

### 4. Diffusion / flow 动作采样随机化

当前模型推理不是完全确定的闭式策略。无论 `model.policy_mode=diffusion` 还是
`model.policy_mode=flow_matching`，动作轨迹都会先从高斯噪声初始化：

```python
sampled_actions = torch.randn(
    (batch_size, self.args.pred_horizon, self.action_dim), device=self.device)
```

因此 diffusion/flow 采样是一个**实际存在的随机源**，不是关闭或固定的随机化。它受
`ADA_MANIP_SEED` 初始化的 PyTorch 随机种子影响；在同一次 run 中，不同 episode 会继续消耗
同一个随机数序列，所以即使同一个 env 的机构状态和 prompt 都固定，不同 episode 的动作采样噪声
也可能不同。

需要注意的是，seed 固定通常可以让同配置的整次 run 更可复现，但 GPU kernel、IsaacGym 物理仿真
和接触求解仍可能带来非逐帧完全一致的细节差异。

### 5. Asker 后端随机性

不同 asker 后端的随机性不同：

| asker platform | 随机性 |
|---|---|
| `ground-truth` | 基于 `frozen_clock_wise` 返回真实 chain，没有模型随机性 |
| `codex-cli` | 外部模型推理可能存在非确定性；`ADA_MANIP_SEED` 不能保证 Codex 输出完全复现 |
| `gemini` / `claude-cli` | 同理，外部模型和服务端实现可能带来非确定性 |

因此 `ADA_MANIP_SEED` 能控制本地 Python/NumPy/PyTorch 以及本地策略选择的随机源，但不能保证
外部 LLM 后端逐次输出完全一致。

### 6. 当前默认关闭或固定的随机化

以下配置会让一部分潜在随机性在当前推理实验中保持固定：

| 随机源 | 当前配置 | 影响 |
|---|---|---|
| asset 顺序 | `env.asset.randomAsset: false`, `env.asset.StartID: 0` | env id 对应的 microwave asset 固定 |
| 初始物体 pose | `env.randomPose: 0.0` | 不引入额外物体位姿随机扰动 |
| reset noise | `resetPositionNoise: 0.0`, `resetRotationNoise: 0.0`, `resetDofPosRandomInterval: 0.0`, `resetDofVelRandomInterval: 0.0` | reset 后 franka/物体 DOF 状态基本固定 |

这意味着改变 `ADA_MANIP_SEED` 不会改变 env id 对应的 asset，也不会改变当前关闭的 pose/reset
随机扰动。某些 env 反复表现为“较晚成功”或出现夹爪松开、门回关等现象，可能来自固定 asset 几何、
当前策略轨迹和仿真物理，而不是 seed 没有生效。

### 7. 如何确认 seed 是否生效

推荐检查两项：

1. 查看输出目录的 `eval_config.yaml`，确认 `seed` 字段是期望值。
2. 查看 `eval_metrics.json` 中第 1 个 episode 的 `episodes[0].envs[*].clock_wise`。

例如 `env.clockwise=0.5`、`numEnvs=10` 时，不同 seed 通常会得到不同的 `clock_wise` 分布。
如果 `seed` 字段正确且 `clock_wise` 分布随 seed 改变，就说明本地 seed 已经生效。
