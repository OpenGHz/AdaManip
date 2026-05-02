# AdaManip Inference / Evaluation Flow

本文档整理当前 microwave 模型推理流程。示例入口脚本：

```bash
sh third_party/ada_manip/scripts/microwave/eval_microwave_model.sh
```

脚本内部使用仓库根目录的 pixi manifest，并分别启动两个运行环境：

1. `ada-data`：启动 IsaacGym 原生环境，并通过 RPyC 暴露远程环境服务。
2. `ada-manip`：加载策略模型，作为 RPyC client 远程驱动环境完成推理。

## 1. 启动参数

`scripts/microwave/eval_microwave_model.sh` 的默认参数如下：

```bash
TASK_NAME=OpenMicroWave
MANIPULATION_NAME=OpenMicroWaveManipulation
CFG_ENV=cfg/microwave/open_microwave_model.yaml
RPYC_HOST=localhost
RPYC_PORT=18861
SIM_DEVICE=cuda:0
SEED=0
```

可通过环境变量覆盖：

```bash
ADA_MANIP_CFG_ENV=cfg/microwave/open_microwave_model.yaml \
ADA_MANIP_RPYC_HOST=localhost \
ADA_MANIP_RPYC_PORT=18861 \
ADA_MANIP_SIM_DEVICE=cuda:0 \
ADA_MANIP_SEED=0 \
sh third_party/ada_manip/scripts/microwave/eval_microwave_model.sh
```

脚本会根据自身路径定位：

1. `ADA_MANIP_ROOT=third_party/ada_manip`
2. `REPO_ROOT=项目根目录`

然后所有 Python 命令都通过：

```bash
pixi run --manifest-path "$REPO_ROOT/pyproject.toml" -e <env> python run.py ...
```

## 2. 两进程结构

### 2.1 RPyC server：真实仿真环境

脚本先在后台启动 server：

```bash
pixi run --manifest-path "$REPO_ROOT/pyproject.toml" -e ada-data python run.py \
  --task=OpenMicroWave \
  --controller=ModelController \
  --manipulation=OpenMicroWaveManipulation \
  --sim_device=cuda:0 \
  --seed=0 \
  --pipeline=gpu \
  --cfg_env=cfg/microwave/open_microwave_model.yaml \
  --runtime_mode=rpyc-server \
  --rpyc_host=localhost \
  --rpyc_port=18861
```

server 进程执行 `run.py` 后：

1. `get_args()` 走 IsaacGym 参数解析。
2. `load_cfg()` 读取 `cfg/microwave/open_microwave_model.yaml`。
3. `parse_sim_params()` 构造 IsaacGym 仿真参数。
4. `parse_env()` 创建原生 `OpenMicroWave` 环境。
5. 因为 `runtime_mode=rpyc-server`，`run.py` 不创建 manipulation/controller，而是调用 `ipc.service.serve_env()`。

RPyC 服务暴露的关键环境接口包括：

1. `reset()` / `reset_with_kwargs()`
2. `step(actions)`
3. `collect_diff_data()`
4. `collect_rgb_frames()`
5. `get_state_snapshot()`
6. `gripper` / `actions` setter

脚本退出、被中断或 client 结束时，会通过 `trap cleanup` 杀掉后台 server。

注意：当前脚本中 server 端的 `--headless` 是注释状态；真正控制 IsaacGym 窗口的是 server 进程。client 端虽然传了 `--headless`，但 client 只连接远程环境，不创建原生仿真窗口。

### 2.2 RPyC client：模型推理控制器

server 启动后，脚本启动 client：

```bash
pixi run --manifest-path "$REPO_ROOT/pyproject.toml" -e ada-manip python run.py \
  --task=OpenMicroWave \
  --controller=ModelController \
  --manipulation=OpenMicroWaveManipulation \
  --sim_device=cuda:0 \
  --seed=0 \
  --pipeline=gpu \
  --cfg_env=cfg/microwave/open_microwave_model.yaml \
  --runtime_mode=rpyc-client \
  --rpyc_host=localhost \
  --rpyc_port=18861 \
  --headless
```

client 进程执行 `run.py` 后：

1. `get_args()` 使用 client 专用 parser，避免加载 IsaacGym 的 server 参数解析逻辑。
2. `parse_sim_params()` 因为 `runtime_mode=rpyc-client` 返回 `None`。
3. `parse_env()` 返回 `ipc.remote_env.RemoteEnv`，连接到 server。
4. `parse_manipulation()` 创建 `OpenMicroWaveManipulation`，其 `env` 是 `RemoteEnv`。
5. `parse_controller()` 创建 `ModelController`。
6. `ModelController.run()` 调用 `OpenMicroWaveManipulation.diffusion_evaluate()`。

## 3. 模型加载

推理配置来自 `cfg/microwave/open_microwave_model.yaml` 的 `model` 段。当前关键项：

```yaml
model:
  policy_mode: diffusion
  obs_horizon: 2
  action_horizon: 1
  pred_horizon: 4
  num_diffusion_iters: 100
  dof_dim: 0
  action_dim: 10
  input_feat: 3
  feat_dim: 128
  Transformer: False
  diffusion_model_path: logs/microwave/manip/Mar17/13-38-17/ema_nets.pth
  use_language_conditioning: True
  language_input_dim: 512
  language_proj_dim: 128
```

`ModelController` 会把这些配置写入 `diffusion_policy.diffusion_policy_new.argument()`：

1. `ckpt_path` 优先使用 `model.model_path`，否则使用 `model.diffusion_model_path`。
2. `policy_mode` 决定采样方式：
   - `diffusion`：从高斯噪声动作开始，迭代 `num_diffusion_iters` 次去噪。
   - `flow_matching`：从高斯噪声动作开始，按 `flow_sampling_steps` 欧拉积分。
3. `Transformer=False` 时加载 `diffusion_policy_new.DiffusionPolicy`。
4. checkpoint 会通过 `DiffusionPolicy.load_checkpoint()` 加载，并切换到 `eval()`。

如果 `use_language_conditioning=True`，评估阶段会加载语言 embedding bank：

1. 若 `model.language_embedding_dict_path` 显式配置，则读取该路径。
2. 否则读取 checkpoint 同目录下的 `language_embedding_dict.json`。
3. 文件内必须包含二维数组字段 `expanded_minimal_chains`。
4. embedding 维度必须等于 `language_input_dim`。

## 4. 单个 episode 推理流程

`OpenMicroWaveManipulation.diffusion_evaluate()` 是当前 microwave 评估主循环。

### 4.1 Episode 初始化

每个 episode 开始时：

1. 如果启用语言条件，从 `expanded_minimal_chains` 中随机采样一个 chain embedding id。
2. 采样到的 embedding 会复制到所有并行环境，因此同一轮 episode 内所有 env 使用同一个语言条件。
3. 调用 `env.reset(clock_same=False)` 重置远程环境。
4. 将 `env.gripper` 置零。
5. 调用 `env.collect_diff_data()` 获取初始观测。
6. 用 `dataset.dataset.obs_wrapper()` 拆成：
   - `pcs`
   - `env_state = proprioception + dof_state + prev_action`
7. 用初始观测填充长度为 `obs_horizon` 的 `pcs_deque` 和 `env_state_deque`。

当前 microwave 配置中 `dof_dim=0`，所以 `obs_wrapper()` 不拼接任务机构 dof，只使用 proprioception 和上一动作。

### 4.2 多 episode 时的环境状态

每个 episode 都会重新 reset 环境，当前评估代码调用的是：

```python
self.env.reset(clock_same=False)
```

因此同一个 env 槽位在不同 episode 中不会沿用上一轮的门状态。reset 会恢复机器人、微波炉 DOF、gripper、上一动作等动态状态，并重新采样每个环境的 `clock_wise`：

```python
self.clock_wise = torch.tensor(
    np.random.rand(self.env_num) < self.cfg["env"]["clockwise"]
)
```

含义：

1. `clock_wise=1`：该 env 的微波炉门初始锁住，直接拉门无效，需要先按按钮解锁。
2. `clock_wise=0`：该 env 的微波炉门可直接拉开。
3. `clock_same=False`：每个 env 独立采样；同一个 env 在下一个 episode 会重新采样，可能与上一轮不同。
4. 默认 `env.clockwise=0.5` 时，每个 env 每个 episode 都有 50% 概率锁住、50% 概率不锁。

例如 env 1 在第 1 个 episode 中可能是 `clock_wise=1`，必须先按按钮；到第 2 个 episode，env 1 会重新按 `env.clockwise` 采样，可能仍然锁住，也可能变成可直接拉开。

注意：这里重新采样的是机构锁定状态和动态状态；已加载的 asset/env 槽位通常不会在 episode 之间重新换成另一台微波炉。

### 4.3 闭环推理

主循环条件为 `step <= 32`。每次循环：

1. 调用：

```python
action = diffusion.infer_action_with_seg(
    pcs_deque,
    env_state_deque,
    language_embedding=episode_language_embedding,
)
```

2. 模型输出形状为：

```text
(num_envs, pred_horizon, action_dim)
```

3. 只执行前 `action_horizon` 个动作：

```python
action = action[:, :diffusion.args.action_horizon, :]
```

当前配置 `action_horizon=1`，所以每次模型预测 4 步，但环境只执行第 1 步，然后重新采集观测再推理。

### 4.4 单步动作执行

每个被执行的动作包含：

1. `action[..., :3]`：末端位置。
2. `action[..., 3:9]`：6D rotation representation。
3. `action[..., -1]`：夹爪开合 logit / value。

执行时：

1. 将 6D rotation 转为 quaternion。
2. 拼出环境动作 `pre_action = xyz + quat`。
3. 当 `action[..., -1] > 0.5` 时令 gripper 闭合，否则打开。
4. 对同一个 `pre_action` 连续调用 `env.step(pre_action)` 15 次。
5. 将原始 10 维 action 写回 `env.actions`，作为下一帧 `prev_action`。
6. 重新 `collect_diff_data()`，并将新观测追加到两个 deque 中。

因此当前推理是 receding horizon / 闭环控制：模型持续使用最近 `obs_horizon` 帧观测重采样未来动作，但每次只执行 `action_horizon` 步。

## 5. 成功判定与输出

每轮 episode 中，所有并行环境共享同一个模型与语言条件，但分别统计成功：

```python
abs(env.one_dof_tensor[env_id, 0]) > pi / 7
```

某个 env 第一次满足该条件时：

1. `done_flag[env_id] = True`
2. `succ_cnt += 1`
3. 打印 `Env <id> Succeeded`

episode 结束后：

1. `current succ rate = succ_cnt / env.num_envs`
2. `succ_cnt` 清零，进入下一轮 episode。
3. 全部 episode 完成后打印：
   - `Average Success rate`
   - `Success rate std`

当前配置中：

```yaml
task:
  num_episode: 1

env:
  numEnvs: 10
```

因此默认会评估 1 轮、每轮 10 个并行环境。

如果 `env.collectRGBVideo=True`，视频会写到：

```text
eval_data/eval_open_microwave_<policy>_<AssetNum>_eps<num_episode>_clock<clockwise>_<timestamp>/rgb_videos/
```

成功与失败环境会分别移动到 `success/` 和 `failure/` 子目录。若未启用 RGB 视频，当前评估流程主要输出终端日志中的成功率。

## 6. 常见调整点

### 6.1 更换 checkpoint

修改 `cfg/microwave/open_microwave_model.yaml`：

```yaml
model:
  diffusion_model_path: logs/microwave/manip/<run>/ema_nets.pth
```

如果使用 `model_path`，`ModelController` 会优先使用它。

### 6.2 关闭语言条件

若 checkpoint 不包含语言条件模块，或没有对应 embedding 文件：

```yaml
model:
  use_language_conditioning: False
```

若保持开启，则需要保证以下文件存在：

```text
<checkpoint_dir>/language_embedding_dict.json
```

或显式配置：

```yaml
model:
  language_embedding_dict_path: path/to/language_embedding_dict.json
```

### 6.3 修改评估规模

```yaml
task:
  num_episode: 10

env:
  numEnvs: 10
```

总评估环境数约为 `num_episode * numEnvs`。成功率按 episode 先求每轮比例，再对所有 episode 求均值和标准差。

### 6.4 切换 RPyC 端口

如果 `18861` 被占用：

```bash
ADA_MANIP_RPYC_PORT=18862 sh third_party/ada_manip/scripts/microwave/eval_microwave_model.sh
```

server 与 client 必须使用同一个 host/port。

### 6.5 真实 headless 运行

若需要 IsaacGym server 无窗口运行，需要在 `eval_microwave_model.sh` 的 server 命令中启用 `--headless`。client 端的 `--headless` 不会影响 server 端窗口。

## 7. 当前流程摘要

```text
eval_microwave_model.sh
  |
  |-- ada-data / rpyc-server
  |     run.py
  |       -> parse native OpenMicroWave env
  |       -> serve_env(env)
  |
  `-- ada-manip / rpyc-client
        run.py
          -> RemoteEnv(host, port)
          -> OpenMicroWaveManipulation(RemoteEnv)
          -> ModelController
          -> DiffusionPolicy.load_checkpoint()
          -> diffusion_evaluate()
                -> reset remote env
                -> collect obs
                -> sample language embedding if enabled
                -> infer pred_horizon actions
                -> execute action_horizon actions
                -> repeat until step limit
                -> compute success rate
```
