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
