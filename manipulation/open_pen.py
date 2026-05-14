from manipulation.base_manipulation import BaseManipulation
from envs.base_env import BaseEnv
from manipulation.utils.transform import *
from logging import Logger
from typing import Any, Dict, List, Optional
import numpy as np
from dataset.dataset import Experience, Episode_Buffer, obs_wrapper
import os
import collections

class OpenPenManipulation(BaseManipulation) :

    _CHAIN_DIRECT: List[str] = ["向上提起笔盖"]
    _CHAIN_ROTATE_LIFT: List[str] = ["Nx旋转笔盖", "向上提起笔盖"]
    # one_go variant: single rotation step is enough, so the chain drops Nx.
    _CHAIN_ROTATE_LIFT_ONE_GO: List[str] = ["旋转笔盖", "向上提起笔盖"]

    def __init__(self, env : BaseEnv, cfg : dict, logger : Logger) :

        super().__init__(env, cfg, logger)

    # ------------------------------------------------------------------
    # Hooks consumed by BaseManipulation.diffusion_evaluate
    # ------------------------------------------------------------------

    @property
    def _one_go(self) -> bool:
        return bool(self.cfg.get("task", {}).get("one_go", False))

    def language_template_task_name(self) -> str:
        return "pen_one_go" if self._one_go else "pen"

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
        # Env-state-derived optimal chain. ``_pen_intrinsic_n[env_id]`` is
        # the per-env N_min recorded by ``collect_manip_data`` at the
        # cumulative rotation step where ``self.env.open_bottle_stage``
        # first transitioned to True (env-physics fact, deterministic per
        # env state). All shared logic — lookup + warn-on-miss fallback —
        # lives in BaseManipulation.ground_truth_chain_from_intrinsic_n.
        # Grasp-data flow doesn't track rotations (collect_grasp_data never
        # initializes `_pen_intrinsic_n`). When the attribute is absent,
        # silently return the canonical chain instead of going through the
        # Nx machinery (which would warn-banner per env, per episode).
        n_min_list = getattr(self, "_pen_intrinsic_n", None)
        if n_min_list is None:
            return self.canonical_minimal_chain_for_state(state)
        if self._one_go:
            # one_go physics: a single rotation crosses the threshold, so
            # the chain has no Nx prefix. Whether rotation is needed at all
            # is still env-state-dependent (cw=0 envs may start with the
            # cap already loose), so consult intrinsic_n.
            if env_id < len(n_min_list) and n_min_list[env_id] is not None:
                if int(n_min_list[env_id]) == 0:
                    return ["向上提起笔盖"]
                return ["旋转笔盖", "向上提起笔盖"]
            return self.canonical_minimal_chain_for_state(state)
        return self.ground_truth_chain_from_intrinsic_n(
            env_id=env_id,
            state=state,
            n_min_attr="_pen_intrinsic_n",
            rotate_op="旋转笔盖",
            lift_op="向上提起笔盖",
            success_hint="opened the pen cap",
        )

    def concrete_attempt_chain_for_collect(self, env_id: int, state: Dict[str, Any]):
        chains = getattr(self, "_pen_attempt_chains", None)
        statuses = getattr(self, "_pen_stage_statuses", None)
        # Grasp-data flow: see open_bottle.py for full rationale.
        if chains is None:
            return None
        if (
            statuses is not None
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
        self.env.reset()
        pose[:, 2] += self.env.gripper_length*2
        for i in range(30):
            self.env.step(pose)
            print("step_{}".format(i+1))
        # grasp the handle
        pose[:, 2] -= self.env.gripper_length - 0.015
        for i in range(28):
            self.env.step(pose)
            print("step_{}".format(i+1+30))
        self.env.gripper = True
        for i in range(10):
            self.env.step(pose)
            print("step_{}".format(i+1+60))
        rot_quat = torch.tensor([[ 0, 0, -0.1305262, 0.9914449]]*batch_size, device=self.env.device) 
        step_size = 0.015
        
        for i in range(50):
            print("step_{}".format(i))
            handle_q = self.env.part_rigid_body_tensor[:, 3:7]
            
            open_dir = quat_axis(handle_q, axis=1)
            cur_p = self.env.hand_rigid_body_tensor[:, :3]
            cur_q = self.env.hand_rigid_body_tensor[:,3:7]

            pred_p = torch.where(self.env.open_bottle_stage.unsqueeze(1).repeat_interleave(3, dim=-1), 
                                 cur_p + open_dir * step_size, cur_p)

            open_door_flag = self.env.open_bottle_stage.unsqueeze(1).repeat_interleave(4, dim=-1)
            pred_q = torch.where(open_door_flag, cur_q, quat_mul(cur_q, rot_quat))
            pred_pose = torch.cat([pred_p, pred_q], dim=-1).float()
            for j in range(15):
                self.env.step(pred_pose)   
 
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
                
                self.env.actions = action[:, act, :]
                obs = self.env.collect_diff_data()
                pcs, env_state = obs_wrapper(obs)

                pcs_deque.append(pcs)
                env_state_deque.append(env_state)

        hand_pose = self.env.hand_rigid_body_tensor[:,:7]
        self.env.gripper = True
        for i in range(10):
            self.env.step(hand_pose)
        print("grasp done")

    '''
    test model
    '''
    # diffusion_evaluate is provided by BaseManipulation.

    '''
    collect grasp data
    '''
    def collect_grasp_data(self):
        eps_num = self.cfg["task"]["num_episode"]
        ctx = self.collect_setup(role="grasp")
        demo_buffer = Experience()
        np.random.seed(self.cfg['task']['seed'])
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

            # grasp the handle
            pre_pose[:, 2] -= self.env.gripper_length - 0.016
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

            # All envs treated as successful in the grasp demo (positioning policy).
            done_flag = [True] * self.env.num_envs
            print(f"Episode {eps} Succeeded")
            self.collect_episode_end(ctx, eps, done_flag, self.eps_buffer, demo_buffer)

        self.collect_finalize(ctx, demo_buffer)

    '''                     
    collect data
    '''
    def collect_manip_data(self, grasp_net=None):
        eps_num = self.cfg["task"]["num_episode"]
        policy = self.cfg["task"]["policy"]
        max_step = 30 if policy == "adaptive" else 25
        print("policy_{}--max_step_{}--num_eps_{}".format(policy, max_step, eps_num))
        ctx = self.collect_setup(role="manip")
        demo_buffer = Experience()

        for eps in range(eps_num):
            eps_buffer = [Episode_Buffer() for _ in range(self.env.num_envs)]
            done_flag = [False] * self.env.num_envs
            print("eps_{}".format(eps+1))
            self.env.reset()
            self.collect_episode_start(ctx, eps)
            # Per-episode attempt-chain state. Each entry in
            # ``_pen_attempt_chains[env_id]`` is one stage string
            # (e.g. "3x旋转笔盖" or "向上提起笔盖"); ``_pen_stage_statuses``
            # tracks per-stage success/failure (rotate stages = True;
            # lift stages = False unless they caused success).
            self._pen_attempt_chains = [[] for _ in range(self.env.num_envs)]
            self._pen_stage_statuses = [[] for _ in range(self.env.num_envs)]
            # Per-env intrinsic N: cumulative rotation count at which the
            # env's ``open_bottle_stage`` flag first transitions to True.
            # Snapshot of env physics — same env state always yields the
            # same N regardless of demo's eventual rollout. ``ground_truth_chain_for_collect``
            # reads this at episode end.
            self._pen_intrinsic_n = [None] * self.env.num_envs
            self._pen_cum_rot = [0] * self.env.num_envs
            current_op = [None] * self.env.num_envs
            current_count = [0] * self.env.num_envs

            one_go = self._one_go

            def _flush(env_id, success):
                op = current_op[env_id]
                cnt = current_count[env_id]
                if op is None or cnt == 0:
                    return
                if op == "旋转笔盖":
                    # one_go variant: chain is non-Nx, so emit the bare op.
                    stage = "旋转笔盖" if one_go else f"{cnt}x旋转笔盖"
                    self._pen_attempt_chains[env_id].append(stage)
                    self._pen_stage_statuses[env_id].append(True)
                else:
                    self._pen_attempt_chains[env_id].append("向上提起笔盖")
                    self._pen_stage_statuses[env_id].append(bool(success))
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

            # grasp the handle
            pre_pose[:, 2] -= self.env.gripper_length - 0.016
            for i in range(3):
                for j in range(10):
                    self.env.step(pre_pose)

            hand_pose = self.env.hand_rigid_body_tensor[:,:7]

            self.env.gripper = True
            for i in range(10):
                self.env.step(hand_pose)

            '''
            set the env previous action to the current hand pose
            '''
            init_actions = self.action_process(hand_pose)
            self.env.actions = init_actions
            # If any env is already release-ready before any rotation
            # (rare in pen but possible if random_upper happens to be
            # below 0.85x threshold by initialization), record N_min=0.
            for env_id in range(self.env.num_envs):
                if (
                    bool(self.env.open_bottle_stage[env_id].item())
                    and self._pen_intrinsic_n[env_id] is None
                ):
                    self._pen_intrinsic_n[env_id] = 0
            ##############start collect manipulation data############
            open_size = 0.015
            if self._one_go:
                # one_go: a single rotation step should be visually obvious.
                # Use ~60° per step (sin(30°)=0.5, cos(30°)≈0.866) instead of
                # the ~15° step used in the multi-rotation default.
                rot_quat = torch.tensor([ 0, 0, -0.5, 0.8660254], device=self.env.device)
                s_rot_quat = torch.tensor([ 0, 0, 0.5, 0.8660254], device=self.env.device)
            else:
                rot_quat = torch.tensor([ 0, 0, -0.1305262, 0.9914449], device=self.env.device)
                s_rot_quat = torch.tensor([ 0, 0, 0.1305262, 0.9914449], device=self.env.device)
            rotate_dof = self.env.two_dof_tensor[:,0]
            prev_op_for_env = [None] * self.env.num_envs
            step_idx_for_env = [0] * self.env.num_envs
            for t in range(max_step):
                cur_p = hand_pose[:, :3]
                cur_q = hand_pose[:,3:7]
                pre_p = cur_p.clone()
                pre_q = cur_q.clone()
                res_per_env = []
                for i in range(self.env.num_envs):
                    if policy == "succ":
                        res = self.succ_policy(i)
                    elif policy == "adaptive":
                        res = self.ada_policy(i, t, rotate_dof[i])
                    else:
                        raise NotImplementedError
                    res_per_env.append(res)
                    if res == "z":
                        pre_p[i,2] += open_size
                    elif res == "o":
                        pre_q[i] = quat_mul(cur_q[i],rot_quat)
                    elif res == "r":
                        pre_q[i] = quat_mul(cur_q[i], s_rot_quat)

                pred_pose = torch.cat([pre_p, pre_q], dim=-1).float()
                gt_pose = self.action_process(pred_pose)

                for env_id in range(self.env.num_envs):
                    if not done_flag[env_id]:
                        obs = self.env.collect_single_diff_data(env_id)
                        pc, env_state = obs_wrapper(obs)
                        eps_buffer[env_id].add(pc, env_state, gt_pose[env_id])
                        op = "向上提起笔盖" if res_per_env[env_id] == "z" else "旋转笔盖"
                        # Update per-env attempt-chain state machine: a change
                        # in operation closes the current stage (failed,
                        # because we wouldn't be looping if it had succeeded)
                        # and starts a new one.
                        if current_op[env_id] is None:
                            current_op[env_id] = op
                            current_count[env_id] = 1
                        elif current_op[env_id] == op:
                            current_count[env_id] += 1
                        else:
                            _flush(env_id, success=False)
                            current_op[env_id] = op
                            current_count[env_id] = 1
                        step_idx_for_env[env_id] = len(self._pen_attempt_chains[env_id])
                        prev_op_for_env[env_id] = op
                        self.append_frame_label_for(env_id, step_idx_for_env[env_id], op)

                for j in range(15):
                    self.env.step(pred_pose)
                self._record_video_frame()

                self.env.actions = gt_pose
                # Update intrinsic N tracking. Increment per-env cumulative
                # rotation count for envs that just rotated, then snapshot
                # ``open_bottle_stage`` which the env updates inside step()
                # — first time it's True is the env-physics N_min.
                for env_id in range(self.env.num_envs):
                    if res_per_env[env_id] in ("r", "o"):
                        self._pen_cum_rot[env_id] += 1
                    if (
                        bool(self.env.open_bottle_stage[env_id].item())
                        and self._pen_intrinsic_n[env_id] is None
                    ):
                        self._pen_intrinsic_n[env_id] = int(self._pen_cum_rot[env_id])
                # update env end flag
                for env_id in range(self.env.num_envs):
                    if (torch.abs(self.env.one_dof_tensor[env_id, 0]) > 0.04).cpu().item() and not done_flag[env_id]:
                        # Final stage caused success — flush with status=True.
                        _flush(env_id, success=True)
                        done_flag[env_id] = True
                        self._mark_env_video_done(env_id)
                        print(f"Env {env_id} Succeeded")
            self.collect_episode_end(ctx, eps, done_flag, eps_buffer, demo_buffer)

        self.collect_finalize(ctx, demo_buffer)

    def succ_policy(self, env_id):
        clock_wise = self.env.clock_wise[env_id]
        open_flag = self.env.open_bottle_stage[env_id]
        if open_flag:
            return 'z'
        else:
            if clock_wise:
                return "r"
            else:
                return "o"
    
    def ada_policy(self, env_id, t, dof):
        clock_wise = self.env.clock_wise[env_id]
        open_flag = self.env.open_bottle_stage[env_id]
        if t == 0:
            # First step is sampled from {lift, rotate}; rotate direction follows task orientation.
            if np.random.rand() < 0.5:
                action = 'z'
            else:
                action = 'r' if clock_wise else 'o'
            self.env.action_chosen[env_id, t] = action
            return action
        elif abs(dof) < self.env.try_range and not open_flag:
            # Cap still locked and not yet rotated past try_range —
            # forced rotate in the cw-correct direction. Skip when the
            # cap is already unlocked (cw=0 envs after the env init sets
            # open_bottle_stage True) so the lift path can stay 1-stage.
            if clock_wise:
                self.env.action_chosen[env_id,t] = "r"
                return "r"
            else:
                self.env.action_chosen[env_id,t] = "o"
                return "o"
        else:
            if self.env.action_chosen[env_id,t-1] == "z":
                if open_flag:
                    action = 'z'
                    self.env.action_chosen[env_id, t] = action
                    return action
                else:
                    if clock_wise:
                        self.env.action_chosen[env_id,t] = "r"
                        return "r"
                    else:
                        self.env.action_chosen[env_id,t] = "o"
                        return "o"
            else:
                prob = np.random.rand()
                if prob < 11/20:
                    action = 'z'
                    self.env.action_chosen[env_id, t] = action
                    return action
                else:
                    if clock_wise:
                        self.env.action_chosen[env_id,t] = "r"
                        return "r"
                    else:
                        self.env.action_chosen[env_id,t] = "o"
                        return "o"