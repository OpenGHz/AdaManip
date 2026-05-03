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

成功与失败环境会分别移动到 `success/` 和 `failure/` 子目录。无论是否启用 RGB 视频，当前评估都会写出 `eval_metrics.json`。

### 5.1 推理输出文件与日志

当前 `diffusion_evaluate()` 不会写出 zarr 或 checkpoint；评估指标会保存为 JSON，推理配置会单独保存为 YAML，视频文件取决于 `env.collectRGBVideo`。

默认配置中：

```yaml
env:
  collectRGBVideo: False
```

此时仍会创建一次评估 run 目录，目录中包含指标文件和推理配置快照：

```text
eval_data/
  eval_open_microwave_<policy>_<AssetNum>_eps<num_episode>_clock<clockwise>_<YYYYmmdd-HHMMSS>/
    eval_config.yaml
    eval_metrics.json
```

`eval_config.yaml` 是本次推理使用的解析后配置快照，包含 `task.adaptive_language.asker` 等 asker 相关配置，以及运行时写入的 `seed`、`headless`、`env.asset.StartID` 等字段。复现实验时应优先使用该文件，而不是依赖 `eval_metrics.json` 中的配置摘要。

`eval_metrics.json` 会在评估开始时创建，并在每个 episode 结束后覆盖更新一次；正常结束时 `overall.status` 会从 `running` 更新为 `completed`。因此即使中途停止，也通常能看到已经完成 episode 的部分结果。

终端日志仍会打印：

1. 加载的 `language_embedding_dict.json` 路径和 embedding bank 大小。
2. 非自适应模式下每个 episode 采样到的 `language embedding id`。
3. 每个成功 env 的 `Env <id> Succeeded`。
4. 每个 episode 的 `current succ rate`。
5. 全部 episode 结束后的 `Average Success rate` 和 `Success rate std`。
6. 自适应模式下额外的 per-env chain id、`frozen_clock_wise`、asker 判定、锁定状态和 sweep 次数。

如需把这些日志保存成文件，需要在运行脚本时自行重定向或使用 `tee`，例如：

```bash
sh third_party/ada_manip/scripts/microwave/eval_microwave_model.sh 2>&1 | tee eval_microwave.log
```

当 `env.collectRGBVideo=True` 时，同一个 run 目录下会额外包含 `rgb_videos/`：

```text
eval_data/
  eval_open_microwave_<policy>_<AssetNum>_eps<num_episode>_clock<clockwise>_<YYYYmmdd-HHMMSS>/
    eval_config.yaml
    eval_metrics.json
    rgb_videos/
```

其中 `<policy>`、`<AssetNum>`、`<num_episode>`、`<clockwise>` 来自当前 cfg；最后的时间戳由推理开始时的本地时间生成，用于避免覆盖旧评估结果。

#### 5.1.1 `eval_metrics.json`

该文件用于对比不同配置下的推理效果和耗时，顶层结构如下：

```json
{
  "schema_version": "v1",
  "run_dir": "./eval_data/eval_open_microwave_adaptive_10_eps4_clock0.5_20260502-120000",
  "config_path": "./eval_data/eval_open_microwave_adaptive_10_eps4_clock0.5_20260502-120000/eval_config.yaml",
  "started_at": "2026-05-02T12:00:00+0800",
  "finished_at": "2026-05-02T12:03:10+0800",
  "episodes": [],
  "overall": {}
}
```

`eval_metrics.json` 不再内嵌配置内容；`config_path` 指向同目录下的 `eval_config.yaml`。这样配置保持完整，尤其是 asker 平台、模型、prompt 风格、抽帧参数、CLI 参数等字段不会因为指标摘要遗漏而影响复现。

`episodes` 是逐 episode 列表。每条 episode 记录包含：

1. `episode_id` / `episode_number`。
2. `started_at` / `finished_at`。
3. `elapsed_sec`：该 episode 总耗时；自适应模式下包含 asker 调用和视频重分类耗时。
4. `rollout_elapsed_sec`：只包含环境 rollout 和策略推理耗时，不包含 episode 后的 asker 处理。
5. `success_count`、`num_envs`、`success_rate`。
6. `sampled_language_id`：非自适应模式下本 episode 广播到所有 env 的 chain id。
7. `per_env_language_ids`：自适应模式下每个 env 实际使用的 chain id。
8. `envs`：该 episode 内每个 env 的明细。

`episodes[*].envs` 中每条 env 记录包含：

1. `env_id`。
2. `episode_id`。
3. `success`。
4. `success_step`：第一次达到成功阈值时的闭环推理步数；失败时为 `null`。
5. `time_to_success_sec`：从 episode 开始到该 env 首次成功的耗时；失败时为 `null`。
6. `episode_elapsed_sec` / `rollout_elapsed_sec`：该 env 所在 episode 的总耗时和 rollout 耗时。
7. `clock_wise`：该 env 本 episode 的机构锁定状态。
8. `ground_truth_chain_id`：由当前 env 的真实机构状态得到的正确 chain id。microwave 任务中，`clock_wise=0` 时为 `0`（`拉门`），`clock_wise=1` 时为 `1`（`按按钮 -> 拉门`）。
9. `final_open_dof`：episode 结束时门关节开度。
10. `language_chain_id`：该 env 本 episode 实际使用的语言 chain id。
11. `adaptive`：仅自适应模式下存在，记录 asker 是否跳过、asker 返回结果、锁定状态和尝试历史。

`adaptive` 字段包含：

1. `skipped`：是否跳过本 episode 的 asker 调用。`true` 表示该 env 在之前 episode 已经锁定 prompt，本 episode 直接复用。
2. `reason`：跳过原因；当前主要为 `already_locked`。仅 `skipped=true` 时存在。
3. `asker_success`：asker 是否认为本次视频/轨迹成功得到可用 prompt。仅实际调用 asker 时存在。
4. `asker_chain_id`：asker 返回的 chain id；当 asker 失败或未返回可识别 chain 时为 `null`。
5. `current_chain_id`：本 episode rollout 时实际输入策略的 chain id。
6. `locked_chain_id`：该 env 当前已经锁定、后续 episode 会复用的 chain id；尚未锁定时为 `null`。
7. `tried_chain_ids`：该 env 当前 sweep 中已经尝试过的 chain id 列表。
8. `sweep_count`：chain id 全部尝试后重置 `tried_chain_ids` 的次数；达到 `max_retry_rounds` 后会强制锁定 fallback chain。
9. `video_path`：传给 asker 的视频路径；未启用视频、文件不存在或跳过 asker 时可能为 `null` 或不存在。

`overall` 是整体汇总：

1. `status`：`running` 或 `completed`。
2. `completed_episodes`。
3. `total_trials = completed_episodes * num_envs`。
4. `total_successes`。
5. `success_rate = total_successes / total_trials`。
6. `mean_episode_success_rate` / `std_episode_success_rate`。
7. `total_elapsed_sec`。
8. `mean_episode_elapsed_sec`。
9. `mean_rollout_elapsed_sec`。
10. `asker_prompt_prediction_count`：实际调用 asker 且返回了可识别 chain id 的预测次数。
11. `asker_prompt_correct_count`：`asker_chain_id == ground_truth_chain_id` 的次数。
12. `asker_prompt_accuracy`：整体 asker prompt 准确率，即 `asker_prompt_correct_count / asker_prompt_prediction_count`；没有可统计预测时为 `null`。
13. `per_env`：每个 env 跨 episode 的汇总。

`overall.per_env[*]` 中除成功率和耗时字段外，还包含 asker prompt 正确性字段：

1. `asker_prompt_prediction_count`：该 env 实际调用 asker 且返回了可识别 chain id 的次数。
2. `asker_prompt_correct_count`：该 env 中 `asker_chain_id == ground_truth_chain_id` 的次数。
3. `asker_prompt_accuracy`：该 env 的 prompt 预测准确率；没有可统计预测时为 `null`。
4. `asker_prompt_correct`：该 env 最近一次可统计 asker 预测是否正确；没有可统计预测时为 `null`。
5. `last_asker_chain_id`：该 env 最近一次可统计 asker 预测的 chain id。
6. `last_ground_truth_chain_id`：该 env 最近一次可统计预测对应的 ground-truth chain id。

#### 5.1.2 非自适应模式的视频文件

非自适应模式下，每个 episode 结束后，视频按 env 自身成功判定移动到 `success/` 或 `failure/`：

```text
rgb_videos/
  success/
    episode_0000/
      env_00_cam_00.mp4
      env_01_cam_00.mp4
  failure/
    episode_0000/
      env_02_cam_00.mp4
```

每个 mp4 文件记录一个 episode 中一个 env、一个 camera 的 rollout 画面：

1. 文件名中的 `env_<id>` 对应并行环境编号。
2. 文件名中的 `cam_<id>` 对应 `env.rgbVideo.cameraIds` 选择的相机编号；若 `cameraIds` 为空，fixed/video camera 会默认记录所有可用固定相机。
3. 视频包含 episode 初始帧，以及每次执行一个 `action_horizon` 动作后记录的一帧。
4. `success/` 与 `failure/` 分类依据是 `abs(env.one_dof_tensor[env_id, 0]) > pi / 7`。

#### 5.1.3 自适应模式的视频文件

当 `task.adaptive_language.enable=True` 且 `env.collectRGBVideo=True` 时，视频还会作为 asker 输入。为方便 asker 读取，episode 结束后视频会先保存在扁平目录：

```text
rgb_videos/
  episode_0000/
    env_00_cam_00.mp4
    env_01_cam_00.mp4
```

asker 调用结束后，如果 `task.adaptive_language.asker.recategorize_videos=True`，这些文件会再迁移到：

```text
rgb_videos/
  success/
    episode_0000/
      env_00_cam_00.mp4
  failure/
    episode_0000/
      env_01_cam_00.mp4
```

自适应模式下的 `success/` / `failure/` 视频分类用于辅助查看 asker 结果：已锁定 chain 的 env 会归入 `success`；未锁定时通常按 env 的实际 `done_flag` 归类。`succ_rate` 的统计仍始终使用 env 自身的开门阈值，不使用视频目录分类。

如果 `recategorize_videos=False`，视频保持在 `rgb_videos/episode_<eps>/` 扁平目录中，便于后续手动核查。

#### 5.1.4 当前不会单独落盘的信息

以下原始细节当前只在内存或终端日志中，不会自动另存为独立文件：

1. 每一步模型输出的完整 10 维 action 轨迹。
2. 每帧点云或低维观测。
3. asker 使用的抽帧图像序列。
4. 终端日志全文。

自适应模式会在内存里收集每个 env 的 10 维 action 轨迹，并把它作为 trajectory context 传给 asker；该数组当前不单独保存为 `.npy`、`.json` 或 `.csv` 文件。

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
                -> update eval_metrics.json
```

## 8. 自适应语言条件评估循环（adaptive language-conditioning eval loop）

### 8.1 动机

默认评估流程在每个 episode 开始时随机采样一个 chain embedding id 并广播到所有 env，并通过
`env.reset(clock_same=False)` 让每个 env 重新随机化 `clock_wise`，因此「同一个 env 在哪种语言条件
下能稳定打开微波炉」这条信息无法跨 episode 复用。

自适应模式开启后，每个 env 跨 episode 维护一份独立的状态机：第一次 episode 仍随机抽样
`clock_wise`，但 chain id 不再随机，而是优先选择最有助于推断真实环境状态的语言条件，并把当次 RGB
视频 + 实时 10 维 action 轨迹送进 asker 判定 `(success, chain_id)`。从第二个 episode 起：

- 该 env 的 `clock_wise` 被冻结为 episode 1 的值；
- 若 asker 此前判定成功并返回了一个 chain id，则锁定该 id，后续不再调用 asker；
- 若仍未锁定，从推理优先级列表中选择下一个尚未尝试过的 id；当全部尝试过仍未成功时，重置已尝试集合
  并从优先级列表开头重新尝试。

整个流程的目的是评估：在 asker 给出最优语言条件后，对同一 env 状态后续任务能否更快、更稳定地完成。

### 8.2 cfg 开关

在 `cfg/microwave/open_microwave_model.yaml` 的 `task:` 段下新增 `adaptive_language` 子树（默认
`enable: false`，与原有行为完全一致）：

```yaml
task:
  ...
  adaptive_language:
    enable: false
    max_retry_rounds: 3
    asker:
      platform: ground-truth        # ground-truth | codex-cli | claude-cli | gemini | foxcode | ""
      model: gpt-5.5
      mock: false
      prompt_style: structured
      check_success: together
      frame_max_count: 12
      frame_stride: 1
      camera_id: 1
      trajectory_context: true
      trajectory_representation: delta
      trajectory_sample_points: 0
      recategorize_videos: true
      lock_on_env_success: false
      claude_cli_*:  ...
      codex_cli_*:   ...
      gemini_*:      ...
```

字段语义沿用 [scripts/eval_video2prompt.py](../../scripts/eval_video2prompt.py) 同名参数。

`platform: ground-truth` 走 `Video2PromptGroundTruth`，根据该 env 冻结后的 `clock_wise` 直接给出
`["按按钮","拉门"]`（`clock_wise=1`）或 `["拉门"]`（`clock_wise=0`），并把 env 自身的 `done_flag`
作为 success 信号。这种模式不调用任何 LLM，常用作冒烟测试。

`platform: codex-cli` / `claude-cli` / `gemini` / `foxcode` 接入 `try_to_remember.experience_abstraction.video2prompt.Video2Prompt`，行为与 `eval_video2prompt.py` 等价：用
`stats.phases` + 通用化后的 `additional_prompt` 让模型从轨迹和帧序列中推断 chain id。`try_to_remember`
已经在 `ada-manip` 这个 pixi 环境里以 editable 方式安装。

### 8.3 状态机与每条 episode 的执行

chain 的推理优先级由 [manipulation/language_chain_utils.py](../manipulation/language_chain_utils.py)
中的通用函数计算，不包含 microwave 专用判断。函数输入应是已经展开后的 `expanded_minimal_chains`；
也就是说，`language_template.json` 里的抽象 `Nx...` 需要先在数据侧展开为 `1x...`、`2x...` 等具体
stage。测试脚本 [tests/show_language_chain_reasoning_examples.py](../tests/show_language_chain_reasoning_examples.py)
会为 `Nx` 任务构造少量示例展开，方便人工检查。

`infer_attempt_chain(language_chain, ground_truth_chain)` 的实现原理：

1. 先对输入 chain 做规范化，去掉空 stage，并保留原始 stage 文本作为最终输出形式。
2. 为了判断“某个语言条件是否已经覆盖真实最小链”，会临时把具体重复 stage 展开到原子操作层面，例如 `2x旋转瓶盖` 会展开为 `["旋转瓶盖", "旋转瓶盖"]`。
3. 在原子操作层面检查 `ground_truth_chain` 是否是 `language_chain` 的有序子序列。这里使用子序列而不是前缀，是因为更长的语言条件可能包含真实最小链需要的所有动作；相反方向的动作名不同，例如 `顺时针旋转` 和 `逆时针旋转`，不会被误判为相同操作。
4. 如果子序列检查通过，说明在“模型语言条件遵循能力足够”的假设下，本轮执行 `language_chain` 就能覆盖真实需求，因此 `attempt_chain = language_chain`。
5. 如果子序列检查不通过，说明第一次按 `language_chain` 执行不足以完成真实任务；诊断模型认为之后会追加真实最小恢复链，因此 `attempt_chain = language_chain + ground_truth_chain`。

这个函数只做抽象 chain 推理，不读取 `clock_wise`、几何状态或视频；它表达的是“在某个语言条件下，若真实状态属于某条最小链，理论上会观察到什么完整尝试序列”。

`rank_expanded_minimal_chain_ids(expanded_minimal_chains)` / `sort_expanded_minimal_chains_by_inference_priority(expanded_minimal_chains)` 的实现原理：

1. 把每一条 `expanded_minimal_chains[i]` 轮流当作候选 `language_chain`。
2. 对所有可能的 `ground_truth_chain` 枚举调用 `infer_attempt_chain(language_chain, ground_truth_chain)`。
3. 按得到的 `attempt_chain` 分组：同一个 `attempt_chain` 下可能对应多个 ground-truth chain id。若某个分组只有一个 id，说明只要观察到该 attempt，就能唯一反推出真实状态。
4. 对每个候选 `language_chain` 计算信息量指标：
   - `unique_ground_truth_count/rate`：能被唯一识别的 ground-truth 数量/比例，越大越好。
   - `worst_case_candidate_count`：最坏情况下同一个 attempt 还剩多少个候选 ground-truth，越小越好。
   - `expected_candidate_count`：按 ground-truth 均匀先验加权后的平均候选数，越小越好。
   - `distinct_attempt_count`：这个 language chain 能产生多少种不同 attempt，越多通常越有区分度。
   - `mean_attempt_atomic_length` 和 `language_atomic_length`：作为次级代价项，避免在信息量相同时优先选择明显更长的尝试。
5. 排序 key 固定为：先最大化 `unique_ground_truth_count`，再最小化 `worst_case_candidate_count`，再最小化 `expected_candidate_count`，再最大化 `distinct_attempt_count`，最后依次用 `mean_attempt_atomic_length`、`language_atomic_length` 和原始 chain id 做稳定排序。

因此 microwave 中 `["拉门"]` 会排在 `["按按钮", "拉门"]` 前面：前者在锁住时会形成 `["拉门", "按按钮", "拉门"]`，能区分锁住/未锁；后者无论锁住与否都可能只看到 `["按按钮", "拉门"]`，无法唯一判断锁状态。

每个 env 在 manipulation 上对应一个 `AdaptiveLanguageState`（在
[manipulation/adaptive_language_asker.py](../manipulation/adaptive_language_asker.py)）：

```text
num_chains          : int   # = embedding bank 大小
frozen_clock_wise   : float # episode 1 之后写死
locked_chain_id     : int?  # asker 判定成功时设定，之后跳过 asker
current_chain_id    : int?  # 当前 episode 选中的 id
tried_chain_ids     : set   # 已尝试且未锁定的 id；全部用尽后清空并 sweep_count += 1
sweep_count         : int   # 已经清空过几轮；超过 max_retry_rounds 强制锁定
```

```text
episode k 流程（每个 env 独立判断）：
  if locked_chain_id is not None:           reuse, skip asker
  else:
      cid = pick_next(priority_ids)         # excludes tried_chain_ids; resets if exhausted
      tried_chain_ids.add(cid)
      run rollout, capture video + 10D actions
      success, asker_cid = asker.ask(...)
      if success and asker_cid is not None: locked_chain_id = asker_cid
      elif lock_on_env_success and done_flag: locked_chain_id = current_chain_id
      elif sweep_count >= max_retry_rounds:   force-lock to asker_cid or current_chain_id

reset 路径：
  episode 1:  env.reset(clock_same=False)   # 与默认行为一致
              snapshot env.clock_wise into states
  episode k>1: env.reset(clock_wise_override=tensor([s.frozen_clock_wise for s in states]))
              # _partial_reset 跳过 np.random.rand 直接使用 override；DOF/root 仍按 initial_*_states 重置
```

`clock_wise_override` 仅强制每个 env 的锁定状态；末端执行器/物体 root 状态仍由 `initial_*_states`
重置，因此 episode 之间机器人会重新回到初始位姿，只是面板锁定情况保持一致。

### 8.4 自适应日志解释

自适应初始化时会打印 chain 的推理优先级：

```text
[adaptive] chain inference priority: 0:拉门|unique=1.00|worst=1, 1:按按钮 -> 拉门|unique=0.00|worst=2
```

这里 `unique` 表示如果第一个 episode 使用该 chain，有多少比例的可能 ground-truth 能从观测到的
`attempt_chain` 中唯一确定；`worst` 表示最坏情况下还剩多少个候选 ground-truth。数值越好，该 chain
越优先尝试。实际打印内容会随 `expanded_minimal_chains` 改变。

每个 episode 开始时会打印本轮每个 env 实际使用的 chain id：

```text
[adaptive] eps 1 chain ids per env: env0=T0, env1=T0, env2=T0, ...
```

含义：

1. `eps 1`：当前是第 1 个 episode。
2. `env0=T0`：env 0 本轮使用 chain id `0`。
3. `T` 表示 trial / try，即该 env 还没有锁定 prompt，本轮是一次新尝试；本轮结束后会调用 asker 判断是否要锁定。
4. `L` 表示 locked，即该 env 之前已经锁定 prompt，本轮直接复用锁定的 chain id，并跳过 asker。

例如 `env2=T0` 表示 env 2 当前还未锁定，本轮尝试 chain id `0`；`env4=L1` 表示 env 4 已锁定，继续使用 chain id `1`。

当某个 env 本轮需要调用 asker 时，episode 结束后会打印：

```text
[adaptive] eps 1 env 0 done_flag=True asker_success=True asker_chain_id=0 ground_truth_chain_id=1 current_chain_id=0 tried=[0] sweep=0
```

字段含义：

1. `done_flag=True`：按环境自身成功阈值判断，该 env 本轮成功打开门。
2. `asker_success=True`：asker 认为本轮视频/轨迹能够得到一个可用 prompt。
3. `asker_chain_id=0`：asker 根据本轮视频/轨迹返回的 chain id。
4. `ground_truth_chain_id=1`：由当前 env 的真实机构状态得到的正确 chain id。microwave 任务中，`clock_wise=0` 时为 `0`（`拉门`），`clock_wise=1` 时为 `1`（`按按钮 -> 拉门`）。
5. `current_chain_id=0`：本轮 rollout 实际输入策略的 chain id。它可能与 `asker_chain_id` 不同，因为 `current_chain_id` 是本轮先尝试的 prompt，而 `asker_chain_id` 是事后根据轨迹推断出的正确 prompt。
6. `tried=[0]`：该 env 当前 sweep 中已经尝试过、但在本轮开始前尚未锁定的 chain id 集合。这里表示 env 0 已经尝试过 chain id `0`。
7. `sweep=0`：该 env 已经把所有 chain id 尝试完并重置 `tried` 的次数；`0` 表示还没有完整扫过一轮。

`tried` 的作用是避免同一个 env 在未锁定前反复尝试同一个 chain id。若所有 chain id 都已经在 `tried`
中，下一次 `pick_next()` 会清空 `tried`，`sweep_count += 1`，然后重新从推理优先级列表开头尝试。达到
`max_retry_rounds` 后会强制锁定 fallback chain，避免无限循环。

### 8.5 输出与视频目录布局

自适应模式同样写入 `eval_metrics.json`；每个 env 的 `adaptive` 字段会记录 asker 结果和锁定状态。

终端日志中每个 episode 会打印每个 env 的 `chain_id`（前缀 `L`=已锁定 / `T`=新尝试）、`done_flag`、
asker 的 success/chain_id、当前 sweep 次数；评估结束时打印每个 env 的 `(frozen_clock_wise,
locked_chain_id, tried_chain_ids, sweep_count)` 全景。

视频文件先以扁平布局落到
`eval_data/<run_dir>/rgb_videos/episode_<eps>/env_<id>_cam_<cam>.mp4`，asker 调用完毕后再按
asker 的 success/failure 判定迁移到 `rgb_videos/{success|failure}/episode_<eps>/...`。把
`asker.recategorize_videos` 设为 `false` 可以保留扁平布局以便后续手动核查。

`succ_rate` 仍然基于 env 自身的 `|env.one_dof_tensor[env_id, 0]| > pi/7` 阈值统计，
不受 asker 判定影响——这样可以独立观察「policy 真正打开门的成功率」与「asker 关于轨迹的判读」是否一致。

### 8.6 离线复跑：把推理产物喂回 `scripts/eval_video2prompt.py`

asker 在线给出的 chain id 不一定总是对的。如果想离线调 prompt、换模型或换分辨率重新评估，
可以把 `task.save_inference_data` 设为 `true`，让推理过程把每个 (episode, env)
的视频 + 实时 10 维 action + 任务标签按 `scripts/eval_video2prompt.py` 期望的格式写到
`eval_save_dir`：

```text
eval_data/<run_dir>/
  language_expanded.json        # 由 task_spec 派生：command / operation_set / expanded_minimal_chains / additional_prompt …
  trajectory_language.jsonl     # JSON 数组，每条记录对应一对 (round_idx=eps, env_id)
                                #   - attempt_chain  = 该 env 当次条件化所用 chain（按语言条件分组）
                                #   - minimal_chain  = 由 frozen_clock_wise 派生的标准答案
                                #   - success        = env 自身 DOF 阈值判定
                                #   - frame_range    = 该轨迹在 demo_data.zip data/action 中的 [start, end)
                                #   - 额外字段 clock_wise / language_chain_id_used /
                                #     frozen_clock_wise / locked_chain_id / tried_chain_ids /
                                #     adaptive_asker（asker_success, asker_chain_id, ground_truth_chain_id, …）
                                #     （eval_video2prompt.py 会忽略不认识的字段，不影响兼容性）
  demo_data.zip                 # zarr：data/action (N,10) float32 + meta/episode_ends (N,) int64
  rgb_videos/
    episode_<eps>/env_<id>_cam_<cam>.mp4
  eval_metrics.json
```

启用 `save_inference_data=true` 时会强制使用扁平的 `rgb_videos/episode_<eps>/` 视频布局
（覆盖 `asker.recategorize_videos`），因为 `eval_video2prompt.py` 的 `locate_video()`
就指望这个布局。同时确认 `env.collectRGBVideo: true`，否则 mp4 不会被写出来。

离线复跑：

```bash
pixi run -e ada-manip python scripts/eval_video2prompt.py \
  --data_root third_party/ada_manip/eval_data \
  --data_dir <run_dir 名字> \
  --num_eval 12 \
  --num_envs <numEnvs 与推理一致> \
  --camera_id <与 task.adaptive_language.asker.camera_id 相同> \
  --platform codex-cli \
  --model gpt-5.5 \
  --codex_cli_effort xhigh \
  --prompt_style structured \
  --frame_max_count 12 \
  --trajectory_representation delta \
  --trajectory_sample_points 0 \
  --codex_cli_timeout 900
```

`--platform ground-truth` 走 `Video2PromptGroundTruth`，会以 `success` + `minimal_chain`
直接给出上界。这条路也是验证 dump 完整性最快的烟雾测试。

### 8.7 冒烟测试

**测试 1：保持默认（自适应关闭）**——回归校验。

```bash
sh third_party/ada_manip/scripts/microwave/eval_microwave_model.sh
```
预期：日志和视频目录与今天一致，每个 episode 仍然打印
`episode N language embedding id: X`，没有 `[adaptive] ...` 前缀。

**测试 2：自适应开启 + ground-truth asker**——不调用任何 LLM。把
`task.adaptive_language.enable` 设为 `true`、`task.num_episode` 设为 4、`env.numEnvs` 设为 2，再次运行：

```yaml
task:
  num_episode: 4
  adaptive_language:
    enable: true
    asker:
      platform: ground-truth
env:
  numEnvs: 2
```

预期：
- episode 1 打印每个 env 的 `frozen clock_wise per env`；
- episode 2+ 不再重新随机化 `clock_wise`；
- 微波炉 chain bank 只有 2 项，因此每个 env 在 ≤2 次尝试内即可锁定（前提是策略自身能成功打开）；
- 视频按 asker 判定迁到 `rgb_videos/{success|failure}/episode_<n>/`。

**测试 3：codex-cli asker（真实 LLM 路径）**——把 `asker.platform` 改为 `codex-cli`，模型按需调整，注意单次调用约 30s。

### 8.7 风险与边界

1. asker 的 success 与 env 自己的 `done_flag` 不一致时：默认以 asker 为锁定权威；若希望以 env 阈值
   为准，把 `asker.lock_on_env_success: true` 打开。`succ_rate` 始终用 env 自身的阈值统计。
2. asker 返回的 chain 不在 `expanded_minimal_chains` 中：视为失败，记录警告。
3. 全部 chain id 反复尝试仍未锁定：每跑完一轮自动重置 `tried_chain_ids`，
   `sweep_count` 累计；达到 `max_retry_rounds`（默认 3）后强制锁定为最近一次 asker 给出的 id 或当
   前 episode 使用的 id，避免无限循环。
4. asker 抛异常 / 超时：包裹在 try/except 中，返回 `(False, None)`，循环继续。
5. 视频分类是 `shutil.move`，若失败会保留扁平布局并打印警告。
