# AdaManip Policy: 代码对齐版架构说明

本文档按当前代码实现补全论文未写清的结构细节，重点覆盖：

1. 各模态是否有独立 MLP 编码。
2. 特征维度如何计算。
3. U-Net 的具体层级与通道。
4. 训练/推理的输入输出流。

---

## 1) 模型总览

当前主干是 条件 1D U-Net 噪声预测器 + PointNet2 点云编码器：

- 点云编码器: PointNet2SemSegSSG
- 动作生成器: ConditionalUnet1D
- 训练目标:
  - diffusion 模式: 预测噪声 epsilon
  - flow_matching 模式: 预测速度场

实现入口在 diffusion_policy_new.py。

---

## 2) 观测模态与编码方式

### 2.1 模态组成

每个时刻观测包含：

- 点云 pc
- proprioception
- dof_state
- prev_action

其中低维状态在数据侧直接拼接为 env_state：

env_state = concat(proprioception, dof_state, prev_action)

### 2.2 低维观测维度

在 DiffusionPolicy.build_net 中定义：

low_obs_dim = 9 + 9 + 7 + dof_dim + action_dim

即：

- 9: qpos
- 9: qvel
- 7: hand pose
- dof_dim: 任务可选机构 dof
- action_dim: prev_action 维度（end-effector pose: xyz position + 6D rotation representation [+ gripper pos]）

例子：

- action_dim=9 (no gripper pos), dof_dim=0 时 low_obs_dim=34
- action_dim=10, dof_dim=0 时 low_obs_dim=35

### 2.3 是否有“每个低维模态独立 MLP”

没有。

当前实现里，低维观测不经过独立 MLP；仅做拼接后直接作为条件向量的一部分。

也就是说：

- 点云: 走 PointNet2 编码
- 低维状态: 直接拼接，不单独过 MLP

---

## 3) 点云编码器细节（PointNet2SemSegSSG）

输入形状：

- 原始: (B, T_obs, N, input_feat)
- 默认 input_feat=3

关键实现细节：

- 前向中将点云 reshape 到 (B*T_obs, N, C)
- 然后 repeat 到最后一维 2 倍（用于 xyz 与附加特征通道对齐）
- SA/FP 后接 1x1 Conv 输出 feat_dim
- 每个时刻输出一个全局特征

默认层配置：

SA（Set Abstraction）:

1. npoint=1024, radius=0.1, nsample=32, mlp=[input_feat, 32, 32, 64]
2. npoint=256, radius=0.2, nsample=32, mlp=[64, 64, 64, 128]
3. npoint=64, radius=0.4, nsample=32, mlp=[128, 128, 128, 256]
4. npoint=16, radius=0.8, nsample=32, mlp=[256, 256, 256, 512]

FP（Feature Propagation）:

1. mlp=[128 + input_feat, 128, 128, 128]
2. mlp=[256 + 64, 256, 128]
3. mlp=[256 + 128, 256, 256]
4. mlp=[512 + 256, 256, 256]

输出头:

- Conv1d(128 -> feat_dim, kernel=1)
- BatchNorm1d(feat_dim)
- ReLU

最终视觉特征形状：

- pcs_features: (B, T_obs, feat_dim)

默认 feat_dim=128。

---

## 4) 条件 1D U-Net 细节（ConditionalUnet1D）

### 4.1 输入输出形状

U-Net 输入 sample 形状：

- (B, pred_horizon, action_dim)

网络内部先转置为：

- (B, action_dim, pred_horizon)

输出与输入 sample 同形状。

### 4.2 条件向量构造

obs_features = concat(pcs_features, npose)

- obs_features 形状: (B, T_obs, obs_dim)
- obs_dim = feat_dim + low_obs_dim

然后 flatten 时间维得到：

- global_cond: (B, T_obs * obs_dim)

时间步编码：

- SinusoidalPosEmb(256)
- Linear(256 -> 1024) + Mish
- Linear(1024 -> 256)

U-Net 内总条件维度：

- cond_dim = 256 + T_obs * obs_dim

### 4.3 U-Net 通道与层级

默认参数：

- down_dims = [256, 512, 1024]
- kernel_size = 5
- n_groups = 8
- cond_predict_scale = True

因此通道链：

- input_dim = action_dim
- all_dims = [action_dim, 256, 512, 1024]

Down 路径每层：

- ResidualBlock(cond)
- ResidualBlock(cond)
- Downsample1d (最后一层为 Identity)

Mid 路径：

- 2 x ResidualBlock(cond)

Up 路径每层：

- 与 skip concat
- ResidualBlock(cond)
- ResidualBlock(cond)
- Upsample1d (最后一层为 Identity)

Final：

- Conv1dBlock(start_dim -> start_dim, kernel=5)
- Conv1d(start_dim -> input_dim, kernel=1)

### 4.4 Residual Block 与条件注入

每个 ConditionalResidualBlock1D:

- 主干: 2 个 Conv1dBlock
- Conv1dBlock = Conv1d + GroupNorm + Mish
- 残差支路: 1x1 Conv 或 Identity

条件注入（FiLM）：

- cond_encoder: Mish + Linear(cond_dim -> 2*out_channels) + reshape
- 对中间特征做 scale/bias 调制

---

## 5) 训练与采样流程

### 5.1 Diffusion 模式

训练：

1. 对真值动作加噪得到 noisy_actions
2. U-Net 预测噪声
3. MSE(noise_pred, noise)

调度器：

- DDPM 或 DDIM
- beta_schedule = squaredcos_cap_v2
- num_train_timesteps = num_diffusion_iters（默认 100）

推理：

- 从高斯噪声动作开始
- 迭代 num_diffusion_iters 次去噪
- 得到 pred_horizon 长度动作序列

### 5.2 Flow Matching 模式

训练：

1. 采样 source_actions ~ N(0, I)
2. 在 source 与 target action 之间线性插值
3. U-Net 预测流场
4. MSE(flow_pred, target_flow)

推理：

- 从噪声动作出发
- 欧拉积分 step_count=flow_sampling_steps
- 每步用 U-Net 预测速度场推进

---

## 6) 输入输出流（简图）

### 6.1 前向图（训练与推理共用主干）

```text
pc(B,T,N,C) ---> PointNet2SemSegSSG ---> pc_feat(B,T,feat_dim)
env_state(B,T,low_obs_dim) -------------------------------+
                                                         |
                          concat over feature dim        v
                         obs_feat(B,T,obs_dim) ---> flatten ---> global_cond(B,T*obs_dim)

sample action trajectory x(B,H,action_dim) + timestep ---> ConditionalUnet1D ---> pred(B,H,action_dim)
```

说明：

- H = pred_horizon
- 在 diffusion 下 pred 表示噪声
- 在 flow_matching 下 pred 表示速度场

### 6.2 闭环执行图（以默认 action_horizon=1）

```text
观测窗口 (T_obs帧)
    -> 采样得到未来 H 步动作
    -> 只执行前 action_horizon 步
    -> 更新观测窗口
    -> 重复
```

---

## 7) 代码与论文可能不一致的关键点

1. 论文常见写法是“多模态各自编码后融合”，但当前代码里低维状态没有独立 MLP。
2. U-Net 是 1D 时序卷积 U-Net，不是 3D U-Net。
3. 观测窗口、动作维度、是否含 gripper 取决于具体任务配置 action_dim/numActions。
4. 代码支持 diffusion 与 flow_matching 两种训练/采样模式。

---

## 8) 可直接核对的默认参数（代码）

- obs_horizon = 2
- pred_horizon = 4
- action_horizon = 1
- num_diffusion_iters = 100
- feat_dim = 128
- input_feat = 3
- action_dim: 任务相关（如 9 或 10）

以上参数由具体任务 yaml 和训练脚本共同决定。
