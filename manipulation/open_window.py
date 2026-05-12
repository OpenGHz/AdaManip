from manipulation.base_manipulation import BaseManipulation
from envs.base_env import BaseEnv
from manipulation.utils.transform import *
from logging import Logger
from typing import Any, Dict, List, Optional
import numpy as np
from dataset.dataset import Experience, Episode_Buffer, obs_wrapper
import os
import collections

class OpenWindowManipulation(BaseManipulation) :

    # The Nx prefix is abstract (literal "Nx") at the canonical level; the
    # demo's ``collect_manip_data`` substitutes a concrete count (e.g. "3x...")
    # via ``ground_truth_chain_for_collect`` and ``concrete_attempt_chain_for_collect``.
    _CHAIN_CW: List[str] = ["Nx顺时针旋转把手", "拉开窗户"]
    _CHAIN_CCW: List[str] = ["Nx逆时针旋转把手", "拉开窗户"]
    # one_go variants: single rotation step is enough, so chains drop Nx.
    _CHAIN_CW_ONE_GO: List[str] = ["顺时针旋转把手", "拉开窗户"]
    _CHAIN_CCW_ONE_GO: List[str] = ["逆时针旋转把手", "拉开窗户"]

    def __init__(self, env : BaseEnv, cfg : dict, logger : Logger) :

        super().__init__(env, cfg, logger)

    # ------------------------------------------------------------------
    # Hooks consumed by BaseManipulation.diffusion_evaluate
    # ------------------------------------------------------------------

    @property
    def _one_go(self) -> bool:
        return bool(self.cfg.get("task", {}).get("one_go", False))

    def language_template_task_name(self) -> str:
        return "window_one_go" if self._one_go else "window"

    def dataset_dir_suffix(self) -> str:
        return "clock" + str(self.cfg["env"]["clockwise"])

    def task_success_for_env(self, env_id: int) -> bool:
        return bool(
            (torch.abs(self.env.one_dof_tensor[env_id, 0]) > np.pi / 6).cpu().item()
        )

    def capture_per_env_episode_state(self) -> List[Dict[str, Any]]:
        clock_wise_values = self.env.clock_wise.detach().cpu().numpy().tolist()
        return [{"clock_wise": float(value)} for value in clock_wise_values]

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
        is_cw = int(round(float(cw))) == 1
        if self._one_go:
            return list(self._CHAIN_CW_ONE_GO if is_cw else self._CHAIN_CCW_ONE_GO)
        return list(self._CHAIN_CW if is_cw else self._CHAIN_CCW)

    def ground_truth_chain_for_collect(
        self, env_id: int, state: Dict[str, Any]
    ) -> Optional[List[str]]:
        # Env-state-derived optimal chain. Counts CORRECT-direction rotations
        # the env physically required to flip ``open_bottle_stage`` to True.
        cw = state.get("clock_wise") if state else None
        if cw is None:
            return None
        rotate_op = "顺时针旋转把手" if int(round(float(cw))) == 1 else "逆时针旋转把手"
        if self._one_go:
            n_min_list = getattr(self, "_window_intrinsic_n", None)
            if (
                n_min_list is not None
                and env_id < len(n_min_list)
                and n_min_list[env_id] is not None
            ):
                # window's bank has no zero-rotation chain — even when
                # ``intrinsic_n == 0`` we still emit the direction-keyed
                # rotate→pull chain so trajectory mc_id resolves cleanly.
                return [rotate_op, "拉开窗户"]
            return self.canonical_minimal_chain_for_state(state)
        return self.ground_truth_chain_from_intrinsic_n(
            env_id=env_id,
            state=state,
            n_min_attr="_window_intrinsic_n",
            rotate_op=rotate_op,
            lift_op="拉开窗户",
            success_hint="opened the window",
        )

    def concrete_attempt_chain_for_collect(self, env_id: int, state: Dict[str, Any]):
        # See open_door.py for rationale (captures wrong-direction t=0
        # picks as failed stages in attempt_chain).
        chains = getattr(self, "_window_attempt_chains", None)
        statuses = getattr(self, "_window_stage_statuses", None)
        if (
            chains is not None and statuses is not None
            and env_id < len(chains) and chains[env_id]
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
    def test_env(self, pose, eval=False):
        batch_size = pose.shape[0]
        pose[:,2] += 0.01
        pose[:,0] += self.env.gripper_length*2
        # move to the handle
        self.env.reset()
        for i in range(30):
            self.env.step(pose)
        
        # grasp the handle
        pose[:, 0] -= self.env.gripper_length + 0.012
        for i in range(30):
            self.env.step(pose)
        self.env.gripper = True
        for i in range(10):
            self.env.step(pose)
        
        down_q = torch.stack(self.env.num_envs * [torch.tensor([0.0, 1.0, 0, 0])]).to(self.env.device).view((self.env.num_envs, 4))
        
        step_size = 0.04
        
        for i in range(15):
            print("step_{}".format(i))
            handle_q = self.env.part_rigid_body_tensor[:, 3:7]
            rotate_dir = quat_axis(handle_q, axis=0) # y-up
            open_dir = quat_axis(handle_q, axis=2)
            cur_p = self.env.hand_rigid_body_tensor[:, :3]
            print(self.env.open_bottle_stage)
            pred_p = torch.where(self.env.open_bottle_stage, cur_p + open_dir * step_size, cur_p + rotate_dir * step_size)
            
            pred_q = quat_mul(handle_q, down_q)
            pred_pose = torch.cat([pred_p, pred_q], dim=-1).float()
            for j in range(10):
                self.env.step(pred_pose)
    
    '''
    grasp net eval
    '''
    def diffusion_eval_grasp(self, grasp_net):
        obs = self.env.collect_diff_data()
        pcs, env_state = obs_wrapper(obs)
        pcs_deque = collections.deque([pcs] * grasp_net.args.obs_horizon, maxlen=grasp_net.args.obs_horizon)
        env_state_deque = collections.deque([env_state] * grasp_net.args.obs_horizon, maxlen=grasp_net.args.obs_horizon)
        step = 0
        act_horizon = 3
        while step < 6:
            action_horizon = act_horizon #min(7 - step, act_horizon)
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
        for eps in range(eps_num):
            self.eps_buffer = [Episode_Buffer() for _ in range(self.env.num_envs)]
            print("eps_{}".format(eps+1))
            self.env.reset()
            self.collect_episode_start(ctx, eps)
            pre_pose = self.env.adjust_hand_pose.clone()
            pre_pose[:,2] += 0.01
            pre_pose[:,0] += self.env.gripper_length*2
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
            pre_pose[:, 0] -= self.env.gripper_length + 0.012
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
            for env_id in range(self.env.num_envs):
                demo_buffer.append(self.eps_buffer[env_id])
            print(f"Episode {eps} Succeeded")
            self.collect_episode_end(ctx, eps, done_flag)

        self.collect_finalize(ctx, demo_buffer)


    '''
    collect data
    '''
    def collect_manip_data(self):
        eps_num = self.cfg["task"]["num_episode"]
        policy = self.cfg["task"]["policy"]
        print(f"policy: {policy}")
        succ_cnt = [0] * self.env.num_envs
        ctx = self.collect_setup(role="manip")
        demo_buffer = Experience()
        for eps in range(eps_num):
            eps_buffer = [Episode_Buffer() for _ in range(self.env.num_envs)]
            done_flag = [False] * self.env.num_envs
            print("eps_{}".format(eps+1))
            self.env.reset()
            self.collect_episode_start(ctx, eps)

            pre_pose = self.env.adjust_hand_pose.clone()
            pre_pose[:,2] += 0.01
            pre_pose[:,0] += self.env.gripper_length*2
            for i in range(3):
                for j in range(10):
                    self.env.step(pre_pose)
                self._record_video_frame()
            pre_pose[:, 0] -= self.env.gripper_length + 0.012
            for i in range(3):
                for j in range(10):
                    self.env.step(pre_pose)
                self._record_video_frame()

            hand_pose = self.env.hand_rigid_body_tensor[:,:7]

            self.env.gripper = True
            for i in range(10):
                self.env.step(hand_pose)
            self._record_video_frame()
            init_actions = self.action_process(hand_pose)
            self.env.actions = init_actions
            ####################start collect manipulation data###################
            max_step = 25 if policy == "adaptive" else 20
            one_go = self._one_go
            # Default rotate_size; one_go uses a bigger step ONLY in the
            # cw-correct direction (where the dof actually has range).
            # Wrong direction stays at default — a wrong-direction stroke
            # would just push against the dof's hard limit, so a big step
            # would disturb the gripper's grasp without producing useful
            # rotation.
            rotate_size_default = 0.04
            rotate_size_big = 0.08
            open_size = 0.03
            hand_pose = self.env.hand_rigid_body_tensor[:,:7]
            down_q = torch.stack(self.env.num_envs * [torch.tensor([0.0, 1.0, 0, 0])]).to(self.env.device).view((self.env.num_envs, 4))
            rotate_dof = self.env.two_dof_tensor[:,0]

            handle_q = self.env.part_rigid_body_tensor[:, 3:7]

            step_idx_for_env = [0] * self.env.num_envs
            res_to_op = {
                "r": "顺时针旋转把手",
                "y": "逆时针旋转把手",
                "x": "拉开窗户",
            }
            self._window_attempt_chains = [[] for _ in range(self.env.num_envs)]
            self._window_stage_statuses = [[] for _ in range(self.env.num_envs)]
            current_op = [None] * self.env.num_envs
            current_count = [0] * self.env.num_envs
            # Per-env intrinsic N tracking — see open_door.py / open_pen.py.
            self._window_intrinsic_n = [None] * self.env.num_envs
            self._window_cum_rot = [0] * self.env.num_envs

            def _stage_status_for_intermediate(env_id, op):
                cw = int(self.env.clock_wise[env_id].item())
                if op == "顺时针旋转把手":
                    return cw == 1
                if op == "逆时针旋转把手":
                    return cw == 0
                return False  # intermediate 拉开窗户 = failed pull

            def _flush_stage(env_id, op, count, success):
                if op is None or count <= 0:
                    return
                if op in ("顺时针旋转把手", "逆时针旋转把手"):
                    stage = op if one_go else f"{count}x{op}"
                    self._window_attempt_chains[env_id].append(stage)
                else:
                    self._window_attempt_chains[env_id].append(op)
                self._window_stage_statuses[env_id].append(bool(success))

            for t in range(max_step):
                cur_p = hand_pose[:, :3]
                pre_p = cur_p.clone()
                y_rotate_dir = quat_axis(handle_q, axis=0) # y-up
                r_rotate_dir = -y_rotate_dir
                open_dir = quat_axis(handle_q, axis=2)
                res_per_env = []
                for i in range(self.env.num_envs):
                    if policy == "succ":
                        res = self.succ_policy(i)
                    elif policy == "adaptive":
                        res = self.ada_policy(i, t, rotate_dof[i])
                    else:
                        raise NotImplementedError
                    res_per_env.append(res)
                    # In one_go: big rotate_size only in the cw-correct
                    # direction ('r' if cw==1; 'y' if cw==0). Wrong direction
                    # gets the default step so it doesn't fight against the
                    # blocked dof limit.
                    cw_i = int(self.env.clock_wise[i].item()) if one_go else None
                    if res == "x":
                        pre_p[i,:] += open_size * open_dir[i].squeeze(0)
                    elif res == "r":
                        size = rotate_size_big if (one_go and cw_i == 1) else rotate_size_default
                        pre_p[i,:] += size * r_rotate_dir[i].squeeze(0)
                    elif res == "y":
                        size = rotate_size_big if (one_go and cw_i == 0) else rotate_size_default
                        pre_p[i,:] += size * y_rotate_dir[i].squeeze(0)

                pred_q = quat_mul(handle_q, down_q)
                pred_pose = torch.cat([pre_p, pred_q], dim=-1).float()
                gt_pose = self.action_process(pred_pose)

                for env_id in range(self.env.num_envs):
                    if not done_flag[env_id]:
                        obs = self.env.collect_single_diff_data(env_id)
                        pc, env_state = obs_wrapper(obs)
                        eps_buffer[env_id].add(pc, env_state, gt_pose[env_id])
                        op = res_to_op.get(res_per_env[env_id])
                        if op is not None:
                            if current_op[env_id] is None:
                                current_op[env_id] = op
                                current_count[env_id] = 1
                            elif current_op[env_id] == op:
                                current_count[env_id] += 1
                            else:
                                _flush_stage(
                                    env_id,
                                    current_op[env_id],
                                    current_count[env_id],
                                    _stage_status_for_intermediate(env_id, current_op[env_id]),
                                )
                                current_op[env_id] = op
                                current_count[env_id] = 1
                            step_idx_for_env[env_id] = len(self._window_attempt_chains[env_id])
                            self.append_frame_label_for(env_id, step_idx_for_env[env_id], op)

                for j in range(15):
                    self.env.step(pred_pose)
                self._record_video_frame()

                self.env.actions = gt_pose
                # Update intrinsic-N tracking — only CORRECT-direction rotations
                # advance dof toward try_range.
                for env_id in range(self.env.num_envs):
                    cw = int(self.env.clock_wise[env_id].item())
                    correct_action = "r" if cw == 1 else "y"
                    if res_per_env[env_id] == correct_action:
                        self._window_cum_rot[env_id] += 1
                    if (
                        bool(self.env.open_bottle_stage[env_id].item())
                        and self._window_intrinsic_n[env_id] is None
                    ):
                        self._window_intrinsic_n[env_id] = int(self._window_cum_rot[env_id])
                for env_id in range(self.env.num_envs):
                    if (torch.abs(self.env.one_dof_tensor[env_id, 0]) > np.pi/6).cpu().item() and not done_flag[env_id]:
                        if current_op[env_id] is not None:
                            _flush_stage(
                                env_id,
                                current_op[env_id],
                                current_count[env_id],
                                True,
                            )
                            current_op[env_id] = None
                            current_count[env_id] = 0
                        demo_buffer.append(eps_buffer[env_id])
                        done_flag[env_id] = True
                        succ_cnt[env_id] += 1
                        print(f"Env {env_id} Succeeded")
            print(succ_cnt)
            self.collect_episode_end(ctx, eps, done_flag)

        self.collect_finalize(ctx, demo_buffer)
    
    def succ_policy(self, env_id):
        clock_wise = self.env.clock_wise[env_id]
        open_flag = self.env.open_bottle_stage[env_id]
        if open_flag:
            return 'x'
        else:
            if clock_wise:
                return "r"
            else:
                return "y"
    
    def ada_policy(self, env_id, t, dof):
        """Policy faithful to ``docs/design/tasks.md`` §5:
            - t=0: random direction.
            - After a failed open trial / wrong-direction rotation: switch
              direction (doc: "if one direction fails, switch to the other").
            - If prev was rotation: randomly sample {Rotate, Open}.
        Uses ``clock_wise`` only to detect "wrong direction" (cheaty
        collection-time oracle), not to bypass the switch-on-failure logic.
        This matters in one_go mode where ``try_range`` is tiny and the old
        ``abs(dof) < try_range`` correction branch was bypassed after one
        rotation, causing wrong-t=0 envs to never recover.
        """
        clock_wise = self.env.clock_wise[env_id]
        open_flag = self.env.open_bottle_stage[env_id]
        correct_action = 'r' if clock_wise else 'y'

        if t == 0:
            action = 'r' if np.random.rand() > 11/20 else 'y'
            self.env.action_chosen[env_id, t] = action
            return action

        prev = self.env.action_chosen[env_id, t-1]

        # "if direction fails, switch" — if last rotation was in the wrong
        # direction (didn't advance dof toward correct unlock), flip it.
        if prev in ('r', 'y') and prev != correct_action and not open_flag:
            action = correct_action
            self.env.action_chosen[env_id, t] = action
            return action

        # Mechanism not yet unlocked — keep rotating in correct direction.
        if not open_flag:
            action = correct_action
            self.env.action_chosen[env_id, t] = action
            return action

        # Mechanism unlocked → try pull. After a rotation, randomly sample
        # {Rotate, Open}; once pulling, keep pulling.
        if prev == 'x':
            action = 'x'
        else:
            action = 'x' if np.random.rand() < 11/20 else correct_action
        self.env.action_chosen[env_id, t] = action
        return action
