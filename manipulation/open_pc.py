from manipulation.base_manipulation import BaseManipulation
from envs.base_env import BaseEnv
from manipulation.utils.transform import *
from logging import Logger
from typing import Any, Dict, List, Optional
import numpy as np
from dataset.dataset import Experience, Episode_Buffer, obs_wrapper
import os
import collections

class OpenPressureCookerManipulation(BaseManipulation) :

    _CHAIN_DIRECT: List[str] = ["向上提起把手"]
    _CHAIN_ROTATE_LIFT: List[str] = ["Nx旋转把手", "向上提起把手"]
    # one_go variant: single rotation step is enough, so the chain drops Nx.
    _CHAIN_ROTATE_LIFT_ONE_GO: List[str] = ["旋转把手", "向上提起把手"]

    def __init__(self, env : BaseEnv, cfg : dict, logger : Logger) :

        super().__init__(env, cfg, logger)

    # ------------------------------------------------------------------
    # Hooks consumed by BaseManipulation.diffusion_evaluate
    # ------------------------------------------------------------------

    @property
    def _one_go(self) -> bool:
        return bool(self.cfg.get("task", {}).get("one_go", False))

    def language_template_task_name(self) -> str:
        return "pressure_cooker_one_go" if self._one_go else "pressure_cooker"

    def dataset_dir_suffix(self) -> str:
        return "clock" + str(self.cfg["env"]["clockwise"])

    def task_success_for_env(self, env_id: int) -> bool:
        return bool(
            (torch.abs(self.env.one_dof_tensor[env_id, 0]) > 0.025).cpu().item()
        )

    def capture_per_env_episode_state(self) -> List[Dict[str, Any]]:
        cw = self.env.clock_wise
        if hasattr(cw, "detach"):
            values = cw.detach().cpu().numpy().tolist()
        else:
            values = list(cw)
        return [{"clock_wise": float(value)} for value in values]

    def apply_frozen_states_to_reset_kwargs(
        self, frozen_states: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        return {
            "clock_wise_override": [
                float(state.get("clock_wise", 0.0)) for state in frozen_states
            ]
        }

    def canonical_minimal_chain_for_state(
        self, state: Dict[str, Any]
    ) -> Optional[List[str]]:
        cw = state.get("clock_wise") if state else None
        if cw is None:
            return None
        if int(round(float(cw))) != 1:
            return list(self._CHAIN_DIRECT)
        return list(self._CHAIN_ROTATE_LIFT_ONE_GO if self._one_go else self._CHAIN_ROTATE_LIFT)

    def ground_truth_chain_for_collect(
        self, env_id: int, state: Dict[str, Any]
    ) -> Optional[List[str]]:
        # See open_pen.py for full rationale.
        # Grasp-data flow doesn't track rotations; silent canonical fallback.
        n_min_list = getattr(self, "_pc_intrinsic_n", None)
        if n_min_list is None:
            return self.canonical_minimal_chain_for_state(state)
        if self._one_go:
            if env_id < len(n_min_list) and n_min_list[env_id] is not None:
                if int(n_min_list[env_id]) == 0:
                    return ["向上提起把手"]
                return ["旋转把手", "向上提起把手"]
            return self.canonical_minimal_chain_for_state(state)
        return self.ground_truth_chain_from_intrinsic_n(
            env_id=env_id,
            state=state,
            n_min_attr="_pc_intrinsic_n",
            rotate_op="旋转把手",
            lift_op="向上提起把手",
            success_hint="lifted the pressure-cooker handle",
        )

    def concrete_attempt_chain_for_collect(self, env_id: int, state: Dict[str, Any]):
        chains = getattr(self, "_pc_attempt_chains", None)
        statuses = getattr(self, "_pc_stage_statuses", None)
        if (
            chains is not None
            and statuses is not None
            and env_id < len(chains)
            and chains[env_id]
        ):
            return list(chains[env_id]), list(statuses[env_id])
        return super().concrete_attempt_chain_for_collect(env_id, state)

    def per_env_extra_log_fields(
        self, env_id: int, episode_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        clock_wise = episode_state.get("clock_wise") if episode_state else None
        return {
            "clock_wise": float(clock_wise) if clock_wise is not None else None,
            "final_one_dof": float(self.env.one_dof_tensor[env_id, 0].item()),
        }

    '''
    test env
    '''
    def test_env(self):
        pose = self.env.adjust_hand_pose.clone()
        batch_size = pose.shape[0]
        # move to the handle
        pose[:, 2] += self.env.gripper_length*2
        self.env.reset()
        for i in range(30):
            self.env.step(pose)
        # grasp the handle
        pose[:, 2] -= self.env.gripper_length + 0.012
        for i in range(28):
            self.env.step(pose)
        self.env.gripper = True
        for i in range(10):
            self.env.step(pose)
        down_q = torch.stack(self.env.num_envs * [torch.tensor([0.5, 0.5, -0.5, 0.5])]).to(self.env.device).view((self.env.num_envs, 4))

        step_size = 0.03
        open_step = 0.01
        for i in range(12):
            print("step_{}".format(i))
            handle_q = self.env.part_rigid_body_tensor[:, 3:7]
            
            open_dir = quat_axis(handle_q, axis=1)
            rotate_dir = quat_axis(handle_q, axis=0)

            cur_p = self.env.hand_rigid_body_tensor[:, :3]
            pred_p = torch.where(self.env.open_bottle_stage.unsqueeze(1).repeat_interleave(3, dim=-1), 
                                 cur_p + open_dir * open_step, cur_p + rotate_dir*step_size)
            pred_q = quat_mul(handle_q, down_q)
            pred_pose = torch.cat([pred_p, pred_q], dim=-1).float()
            for j in range(15):
                self.env.step(pred_pose)   


    # diffusion_evaluate is provided by BaseManipulation.

    '''
    grasp net eval
    '''
    def diffusion_eval_grasp(self, grasp_net):
        obs = self.env.collect_diff_data()
        pcs, env_state = obs_wrapper(obs)
        pcs_deque = collections.deque([pcs] * grasp_net.args.obs_horizon, maxlen=grasp_net.args.obs_horizon)
        env_state_deque = collections.deque([env_state] * grasp_net.args.obs_horizon, maxlen=grasp_net.args.obs_horizon)
        step = 0
        action_horizon = 3
        while step < 6:
            pred_poses = grasp_net.infer_action_with_seg(pcs_deque, env_state_deque).detach()
            action = pred_poses[:, :action_horizon, :]
            step += action_horizon
            
            for act in range(action.shape[1]):
                quat = self.rotate_6d_to_quat(action[:, act, 3:])
                pre_action = torch.cat([action[:, act, :3], quat], dim=-1)
                self.env.get_obj_dof_property_tensor()

                for j in range(10):
                    self.env.step(pre_action)
                # ipdb.set_trace()
                self.env.actions = action[:, act, :]
                obs = self.env.collect_diff_data()
                pcs, env_state = obs_wrapper(obs)

                pcs_deque.append(pcs)
                env_state_deque.append(env_state)

    '''
    grasp net data collect
    '''
    def collect_grasp_data(self):
        eps_num = self.cfg["task"]["num_episode"]
        ctx = self.collect_setup(role="grasp")
        demo_buffer = Experience()
        for eps in range(eps_num):
            self.eps_buffer = [Episode_Buffer() for _ in range(self.env.num_envs)]
            print("eps_{}".format(eps+1))
            self.env.reset()
            self.collect_episode_start(ctx, eps)
            pre_pose = self.env.adjust_hand_pose.clone()
            pre_pose[:, 2] += self.env.gripper_length*2
            for i in range(3):
                obs = self.env.collect_diff_data()
                pc, env_state = obs_wrapper(obs)

                for j in range(10):
                    self.env.step(pre_pose)
                self._record_video_frame()

                gt_action = self.action_process(pre_pose)
                self.env.actions = gt_action.clone()
                for env_id in range(self.env.num_envs):
                    self.eps_buffer[env_id].add(pc[env_id], env_state[env_id], gt_action[env_id])
            pre_pose[:, 2] -= self.env.gripper_length + 0.012
            for i in range(3):
                obs = self.env.collect_diff_data()
                pc, env_state = obs_wrapper(obs)

                for j in range(10):
                    self.env.step(pre_pose)
                self._record_video_frame()

                gt_action = self.action_process(pre_pose)
                self.env.actions = gt_action.clone()
                for env_id in range(self.env.num_envs):
                    self.eps_buffer[env_id].add(pc[env_id], env_state[env_id], gt_action[env_id])

            done_flag = [True] * self.env.num_envs
            print(f"Episode {eps} Succeeded")
            self.collect_episode_end(ctx, eps, done_flag, self.eps_buffer, demo_buffer)

        self.collect_finalize(ctx, demo_buffer)

    '''
    manipulation data collect
    '''
    def collect_manip_data(self):
        eps_num = self.cfg["task"]["num_episode"]
        policy = self.cfg["task"]["policy"]
        max_step = 25 if policy == "adaptive" else 20
        print("policy_{}--max_step_{}--num_eps_{}".format(policy, max_step, eps_num))
        ctx = self.collect_setup(role="manip")
        demo_buffer = Experience()
        succ_cnt = [0] * self.env.num_envs
        for eps in range(eps_num):
            eps_buffer = [Episode_Buffer() for _ in range(self.env.num_envs)]
            done_flag = [False] * self.env.num_envs
            print("eps_{}".format(eps+1))
            self.env.reset()
            self.collect_episode_start(ctx, eps)
            self._pc_attempt_chains = [[] for _ in range(self.env.num_envs)]
            self._pc_stage_statuses = [[] for _ in range(self.env.num_envs)]
            self._pc_intrinsic_n = [None] * self.env.num_envs
            self._pc_cum_rot = [0] * self.env.num_envs
            current_op = [None] * self.env.num_envs
            current_count = [0] * self.env.num_envs

            one_go = self._one_go

            def _flush(env_id, success):
                op = current_op[env_id]
                cnt = current_count[env_id]
                if op is None or cnt == 0:
                    return
                if op == "旋转把手":
                    stage = "旋转把手" if one_go else f"{cnt}x旋转把手"
                    self._pc_attempt_chains[env_id].append(stage)
                    self._pc_stage_statuses[env_id].append(True)
                else:
                    self._pc_attempt_chains[env_id].append("向上提起把手")
                    self._pc_stage_statuses[env_id].append(bool(success))
                current_op[env_id] = None
                current_count[env_id] = 0

            # approach / pre-grasp / grasp-close intentionally do not call
            # `_record_video_frame()` so the manip-side mp4 stays 1:1
            # aligned with the manip zarr.
            pre_pose = self.env.adjust_hand_pose.clone()
            pre_pose[:, 2] += self.env.gripper_length*2
            for i in range(3):
                for j in range(10):
                    self.env.step(pre_pose)
            pre_pose[:, 2] -= self.env.gripper_length + 0.012
            for i in range(3):
                for j in range(10):
                    self.env.step(pre_pose)

            hand_pose = self.env.hand_rigid_body_tensor[:,:7]
            self.env.gripper = True
            for i in range(10):
                self.env.step(hand_pose)
            init_actions = self.action_process(hand_pose)
            self.env.actions = init_actions
            for env_id in range(self.env.num_envs):
                if (
                    bool(self.env.open_bottle_stage[env_id].item())
                    and self._pc_intrinsic_n[env_id] is None
                ):
                    self._pc_intrinsic_n[env_id] = 0
            ####################start collect manipulation data###################
            max_step = 25 if policy == "adaptive" else 20
            # one_go: ~4x linear rotate offset per step so the single rotation
            # is visually obvious.
            step_size = 0.14 if one_go else 0.035
            open_size = 0.015
            hand_pose = self.env.hand_rigid_body_tensor[:,:7]
            handle_quat = self.env.part_rigid_body_tensor[:, 3:7]
            rotate_dir = quat_axis(handle_quat, axis=0)
            down_q = torch.stack(self.env.num_envs * [torch.tensor([0.5, 0.5, -0.5, 0.5])]).to(self.env.device).view((self.env.num_envs, 4))
            rotate_dof = self.env.two_dof_tensor[:,0]
            prev_op_for_env = [None] * self.env.num_envs
            step_idx_for_env = [0] * self.env.num_envs
            for t in range(max_step):
                cur_p = hand_pose[:,:3]
                pre_p = cur_p.clone()

                res_per_env = []
                for i in range(self.env.num_envs):
                    if policy == "succ":
                        res = self.succ_policy(i)
                    elif policy == "adaptive":
                        res = self.ada_policy(i, t, rotate_dof[i])
                    else:
                        raise NotImplementedError
                    res_per_env.append(res)
                    if res == 'z':
                        pre_p[i, 2] += open_size
                    elif res == 'r':
                        pre_p[i] += rotate_dir[i] * step_size
                    else:
                        raise NotImplementedError

                    pre_q = quat_mul(handle_quat, down_q)
                pre_pose = torch.cat([pre_p, pre_q], dim=-1)
                gt_pose = self.action_process(pre_pose)

                for env_id in range(self.env.num_envs):
                    if not done_flag[env_id]:
                        obs = self.env.collect_single_diff_data(env_id)
                        pc, env_state = obs_wrapper(obs)
                        eps_buffer[env_id].add(pc, env_state, gt_pose[env_id])
                        op = "向上提起把手" if res_per_env[env_id] == "z" else "旋转把手"
                        if current_op[env_id] is None:
                            current_op[env_id] = op
                            current_count[env_id] = 1
                        elif current_op[env_id] == op:
                            current_count[env_id] += 1
                        else:
                            _flush(env_id, success=False)
                            current_op[env_id] = op
                            current_count[env_id] = 1
                        step_idx_for_env[env_id] = len(self._pc_attempt_chains[env_id])
                        prev_op_for_env[env_id] = op
                        self.append_frame_label_for(env_id, step_idx_for_env[env_id], op)
                for j in range(15):
                    self.env.step(pre_pose)
                self._record_video_frame()
                self.env.actions = gt_pose
                for env_id in range(self.env.num_envs):
                    if res_per_env[env_id] in ("r", "o"):
                        self._pc_cum_rot[env_id] += 1
                    if (
                        bool(self.env.open_bottle_stage[env_id].item())
                        and self._pc_intrinsic_n[env_id] is None
                    ):
                        self._pc_intrinsic_n[env_id] = int(self._pc_cum_rot[env_id])
                # update done_flag
                for env_id in range(self.env.num_envs):
                    if (torch.abs(self.env.one_dof_tensor[env_id, 0]) > 0.035).cpu().item() and not done_flag[env_id]:
                        _flush(env_id, success=True)
                        done_flag[env_id] = True
                        self._mark_env_video_done(env_id)
                        succ_cnt[env_id] += 1
                        print(f"Env {env_id} Succeeded")
            print(succ_cnt)
            self.collect_episode_end(ctx, eps, done_flag, eps_buffer, demo_buffer)

        self.collect_finalize(ctx, demo_buffer)


    def succ_policy(self, env_id):
        open_flag = self.env.open_bottle_stage[env_id]
        if open_flag:
            return 'z'
        else:
            return 'r'         

    def ada_policy(self, env_id, t, dof):
        open_flag = self.env.open_bottle_stage[env_id]
        if t == 0:
            action = 'z' if np.random.rand() < 0.5 else 'r'
            self.env.action_chosen[env_id, t] = action
            return action
        elif dof < self.env.try_range and not open_flag:
            action = 'r'
            self.env.action_chosen[env_id, t] = action
            return action
        else:
            if self.env.action_chosen[env_id, t-1] == 'z':
                if open_flag:
                    action = 'z'
                    self.env.action_chosen[env_id, t] = action
                    return action
                else:
                    action = 'r'
                    self.env.action_chosen[env_id, t] = action
                    return action
            else:
                action = 'r' if np.random.rand() < 11/20 else 'z'
                self.env.action_chosen[env_id, t] = action
                return action
