"""Microwave-specific manipulation subclass.

All task-agnostic logic (video recorder, eval metrics, language embedding bank,
inference dump, ``diffusion_evaluate`` adaptive loop) lives in
``BaseManipulation``. This module only contains:

* Microwave-specific hook overrides (``capture_per_env_episode_state``,
  ``apply_frozen_states_to_reset_kwargs``, ``canonical_minimal_chain_for_state``,
  ``task_success_for_env``, ``per_env_extra_log_fields``, etc.).
* ``test_env`` and ``collect_manip_data`` — the demonstration policy used to
  generate microwave training data.
"""

from logging import Logger
from typing import Any, Dict, List, Optional

import numpy as np
import random
import torch

from envs.base_env import BaseEnv
from manipulation.base_manipulation import BaseManipulation
from manipulation.utils.transform import *  # noqa: F401,F403  (quat_axis, quat_mul, ...)
from dataset.dataset import Experience, Episode_Buffer


class OpenMicroWaveManipulation(BaseManipulation):

    # Door is locked when ``clock_wise == 1``; canonical answer is the
    # button-then-pull chain. ``clock_wise == 0`` means the door pulls open
    # directly. Used by ``canonical_minimal_chain_for_state`` and the
    # ``concrete_attempt_chain_for_collect`` override.
    _LOCKED_CHAIN: List[str] = ["按按钮", "拉门"]
    _UNLOCKED_CHAIN: List[str] = ["拉门"]

    def __init__(self, env: BaseEnv, cfg: dict, logger: Logger):
        super().__init__(env, cfg, logger)

    # ------------------------------------------------------------------
    # Hooks consumed by BaseManipulation.diffusion_evaluate
    # ------------------------------------------------------------------

    def language_template_task_name(self) -> str:
        return "microwave"

    def dataset_dir_suffix(self) -> str:
        return "clock" + str(self.cfg["env"]["clockwise"])

    def reset_kwargs_initial(self) -> Dict[str, Any]:
        return {"clock_same": False}

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

    def canonical_minimal_chain_for_state(self, state: Dict[str, Any]) -> Optional[List[str]]:
        clock_wise = state.get("clock_wise") if state else None
        if clock_wise is None:
            return None
        return list(
            self._LOCKED_CHAIN if int(round(float(clock_wise))) == 1 else self._UNLOCKED_CHAIN
        )

    def task_success_for_env(self, env_id: int) -> bool:
        return bool(
            (torch.abs(self.env.one_dof_tensor[env_id, 0]) > np.pi / 7).cpu().item()
        )

    def per_env_extra_log_fields(
        self, env_id: int, episode_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        clock_wise = episode_state.get("clock_wise") if episode_state else None
        return {
            "clock_wise": float(clock_wise) if clock_wise is not None else None,
            "final_open_dof": float(self.env.one_dof_tensor[env_id, 0].item()),
        }

    def concrete_attempt_chain_for_collect(self, env_id: int, state: Dict[str, Any]):
        # Microwave's demo policy is deterministic given (start_with_pull, cw):
        #   - start_with_pull=True, cw=0  → ["拉门"]                   succeeds in one step
        #   - start_with_pull=True, cw=1  → ["拉门", "按按钮", "拉门"]  first pull fails, then button, then pull
        #   - start_with_pull=False, cw=0 → ["按按钮", "拉门"]          button is wasted but pull succeeds
        #   - start_with_pull=False, cw=1 → ["按按钮", "拉门"]          button needed, pull succeeds
        # Status convention: rotate/intermediate stages = True; lift stages
        # = True iff the lift caused success. The "succeed in one step"
        # cases mark their single stage True per "successful trajectory's
        # last stage_status must be True".
        start_with_pull = getattr(self, "_microwave_start_with_pull", None)
        if state is None or start_with_pull is None:
            return super().concrete_attempt_chain_for_collect(env_id, state)
        cw = int(round(float(state.get("clock_wise", 0))))
        if start_with_pull:
            if cw == 0:
                return list(self._UNLOCKED_CHAIN), [True]
            else:
                return ["拉门", "按按钮", "拉门"], [False, True, True]
        else:
            return list(self._LOCKED_CHAIN), [True, True]

    def test_env(self, pose, eval=False):
        batch_size = pose.shape[0]
        handle_pos = pose[:, :7].clone()
        button_pos = pose[:, 7:].clone()
        print(handle_pos)
        print(button_pos)
        '''
        two manipulation choice 1.pull handle open door 2.push button then pull handle open door
        '''
        self.env.reset()
        flag = False
        if flag:
            handle_pos[:, 0] += self.env.gripper_length * 2
            for i in range(30):
                self.env.step(handle_pos)
            handle_pos[:, 0] -= self.env.gripper_length + 0.014
            for i in range(30):
                self.env.step(handle_pos)
            self.env.gripper = True
            for i in range(10):
                self.env.step(handle_pos)

            down_q = torch.stack(self.env.num_envs * [torch.tensor([0.7071068, 0.7071068, 0, 0])]).to(self.env.device).view((self.env.num_envs, 4))
            step_size = 0.045
            for i in range(10):
                print("step_{}".format(i))
                handle_q = self.env.rigid_body_tensor[:, 3:7]
                open_dir = quat_axis(handle_q, axis=2)
                cur_p = self.env.hand_rigid_body_tensor[:, :3]
                pred_p = cur_p + open_dir * step_size
                pred_q = quat_mul(handle_q, down_q)
                pred_pose = torch.cat([pred_p, pred_q], dim=-1).float()
                for j in range(15):
                    self.env.step(pred_pose)
        else:
            hand_pose = self.env.hand_rigid_body_tensor[:, :7]
            for i in range(1000):
                self.env.step(hand_pose)
            init_handle_pose = handle_pos.clone()
            init_handle_pose[:, 0] += self.env.gripper_length * 2
            for i in range(2):
                for j in range(15):
                    self.env.step(init_handle_pose)
            init_handle_pose[:, 0] -= self.env.gripper_length + 0.014
            for i in range(2):
                for j in range(15):
                    self.env.step(init_handle_pose)
            self.env.gripper = True
            for i in range(50):
                self.env.step(init_handle_pose)

            self.env.gripper = False
            for i in range(15):
                self.env.step(init_handle_pose)
            # push button
            button_pos[:, 0] += self.env.gripper_length * 2 + 0.012
            for i in range(30):
                self.env.step(button_pos)
            button_pos[:, 0] -= self.env.gripper_length
            for i in range(30):
                self.env.step(button_pos)
            self.env.gripper = True
            for i in range(15):
                self.env.step(button_pos)
            button_pos[:, 0] -= 0.03
            for i in range(15):
                self.env.step(button_pos)
            self.env.gripper = False
            handle_pos[:, 0] += self.env.gripper_length * 2
            for i in range(30):
                self.env.step(handle_pos)
            handle_pos[:, 0] -= self.env.gripper_length + 0.014
            for i in range(30):
                self.env.step(handle_pos)
            self.env.gripper = True
            for i in range(15):
                self.env.step(handle_pos)

            down_q = torch.stack(self.env.num_envs * [torch.tensor([0.7071068, 0.7071068, 0, 0])]).to(self.env.device).view((self.env.num_envs, 4))
            step_size = 0.045
            for i in range(10):
                print("step_{}".format(i))
                handle_q = self.env.rigid_body_tensor[:, 3:7]
                open_dir = quat_axis(handle_q, axis=2)
                cur_p = self.env.hand_rigid_body_tensor[:, :3]
                pred_p = cur_p + open_dir * step_size
                pred_q = quat_mul(handle_q, down_q)
                pred_pose = torch.cat([pred_p, pred_q], dim=-1).float()
                for j in range(15):
                    self.env.step(pred_pose)

    def collect_manip_data(self):
        eps_num = self.cfg["task"]["num_episode"]
        policy = self.cfg["task"]["policy"]
        ctx = self.collect_setup(role=None)
        demo_buffer = Experience()
        for eps in range(eps_num):
            self.eps_buffer = [Episode_Buffer() for _ in range(self.env.num_envs)]
            done_flag = [False] * self.env.num_envs
            print("eps_{}".format(eps + 1))
            self.env.reset()
            self.collect_episode_start(ctx, eps)
            ori_pose = self.env.get_adjust_hand_pose()
            pose = ori_pose.clone()
            handle_pos = pose[:, :7].clone()
            button_pos = pose[:, 7:].clone()
            self.env.gripper = torch.zeros((self.env.num_envs, 1), device=self.env.device)
            # ``_microwave_start_with_pull`` is read by
            # ``concrete_attempt_chain_for_collect`` at episode-end to build
            # the per-env attempt_chain / stage_status.
            self._microwave_start_with_pull = False
            try:
                if policy == "succ":
                    if self.env.clock_wise[0] == 1:
                        self._microwave_start_with_pull = False
                        self.set_current_step(0, "按按钮")
                        button_pos[:, 0] += self.env.gripper_length * 2 + 0.012
                        for i in range(2):
                            self.process_data(button_pos)
                            for j in range(15):
                                self.env.step(button_pos)
                        button_pos[:, 0] -= self.env.gripper_length
                        for i in range(2):
                            self.process_data(button_pos)
                            for j in range(15):
                                self.env.step(button_pos)
                        self.env.gripper = torch.ones((self.env.num_envs, 1), device=self.env.device)
                        self.process_data(button_pos)
                        for j in range(15):
                            self.env.step(button_pos)
                        button_pos[:, 0] -= 0.03
                        for i in range(2):
                            self.process_data(button_pos)
                            for j in range(15):
                                self.env.step(button_pos)
                        self.env.gripper = torch.zeros((self.env.num_envs, 1), device=self.env.device)

                        self.set_current_step(1, "拉门")
                        handle_pos[:, 0] += self.env.gripper_length * 2
                        for i in range(2):
                            self.process_data(handle_pos)
                            for j in range(15):
                                self.env.step(handle_pos)
                        handle_pos[:, 0] -= self.env.gripper_length + 0.014
                        for i in range(2):
                            self.process_data(handle_pos)
                            for j in range(15):
                                self.env.step(handle_pos)
                        self.env.gripper = torch.ones((self.env.num_envs, 1), device=self.env.device)
                        self.process_data(handle_pos)
                        for j in range(15):
                            self.env.step(handle_pos)

                        down_q = torch.stack(self.env.num_envs * [torch.tensor([0.7071068, 0.7071068, 0, 0])]).to(self.env.device).view((self.env.num_envs, 4))
                        step_size = 0.045
                        for i in range(8):
                            handle_q = self.env.rigid_body_tensor[:, 3:7]
                            open_dir = quat_axis(handle_q, axis=2)
                            cur_p = self.env.hand_rigid_body_tensor[:, :3]
                            pred_p = cur_p + open_dir * step_size
                            pred_q = quat_mul(handle_q, down_q)
                            pred_pose = torch.cat([pred_p, pred_q], dim=-1).float()
                            self.process_data(pred_pose)
                            for j in range(15):
                                self.env.step(pred_pose)
                    else:
                        self._microwave_start_with_pull = True
                        self.set_current_step(0, "拉门")
                        handle_pos[:, 0] += self.env.gripper_length * 2
                        for i in range(2):
                            self.process_data(handle_pos)
                            for j in range(15):
                                self.env.step(handle_pos)
                        handle_pos[:, 0] -= self.env.gripper_length + 0.014
                        for i in range(2):
                            self.process_data(handle_pos)
                            for j in range(15):
                                self.env.step(handle_pos)
                        self.env.gripper = torch.ones((self.env.num_envs, 1), device=self.env.device)
                        self.process_data(handle_pos)
                        for i in range(15):
                            self.env.step(handle_pos)

                        down_q = torch.stack(self.env.num_envs * [torch.tensor([0.7071068, 0.7071068, 0, 0])]).to(self.env.device).view((self.env.num_envs, 4))
                        step_size = 0.045
                        for i in range(8):
                            handle_q = self.env.rigid_body_tensor[:, 3:7]
                            open_dir = quat_axis(handle_q, axis=2)
                            cur_p = self.env.hand_rigid_body_tensor[:, :3]
                            pred_p = cur_p + open_dir * step_size
                            pred_q = quat_mul(handle_q, down_q)
                            pred_pose = torch.cat([pred_p, pred_q], dim=-1).float()
                            self.process_data(pred_pose)
                            for j in range(15):
                                self.env.step(pred_pose)
                else:
                    down_q = torch.stack(self.env.num_envs * [torch.tensor([0.7071068, 0.7071068, 0, 0])]).to(self.env.device).view((self.env.num_envs, 4))
                    step_size = 0.045
                    start_with_pull = np.random.rand() < 0.5
                    self._microwave_start_with_pull = start_with_pull

                    if start_with_pull:
                        self.set_current_step(0, "拉门")
                        handle_pos[:, 0] += self.env.gripper_length * 2
                        for i in range(2):
                            self.process_data(handle_pos)
                            for j in range(15):
                                self.env.step(handle_pos)
                        handle_pos[:, 0] -= self.env.gripper_length + 0.014
                        for i in range(2):
                            self.process_data(handle_pos)
                            for j in range(15):
                                self.env.step(handle_pos)
                        self.env.gripper = torch.ones((self.env.num_envs, 1), device=self.env.device)
                        self.process_data(handle_pos)
                        for i in range(15):
                            self.env.step(handle_pos)

                        for i in range(2):
                            handle_q = self.env.rigid_body_tensor[:, 3:7]
                            open_dir = quat_axis(handle_q, axis=2)
                            cur_p = self.env.hand_rigid_body_tensor[:, :3]
                            pred_p = cur_p + open_dir * step_size
                            pred_q = quat_mul(handle_q, down_q)
                            pred_pose = torch.cat([pred_p, pred_q], dim=-1).float()
                            self.process_data(pred_pose)
                            for j in range(15):
                                self.env.step(pred_pose)

                    if start_with_pull and self.env.clock_wise[0] == 0:
                        self.set_current_step(0, "拉门")
                        for i in range(6):
                            handle_q = self.env.rigid_body_tensor[:, 3:7]
                            open_dir = quat_axis(handle_q, axis=2)
                            cur_p = self.env.hand_rigid_body_tensor[:, :3]
                            pred_p = cur_p + open_dir * step_size
                            pred_q = quat_mul(handle_q, down_q)
                            pred_pose = torch.cat([pred_p, pred_q], dim=-1).float()
                            self.process_data(pred_pose)
                            for j in range(15):
                                self.env.step(pred_pose)
                    else:
                        button_step_idx = 1 if start_with_pull else 0
                        pull_step_idx = 2 if start_with_pull else 1

                        self.set_current_step(button_step_idx, "按按钮")
                        self.env.gripper = torch.zeros((self.env.num_envs, 1), device=self.env.device)
                        keep_pose = self.env.hand_rigid_body_tensor.clone()
                        self.process_data(keep_pose)
                        for i in range(15):
                            self.env.step(keep_pose)

                        keep_pose[:, 0] += self.env.gripper_length
                        for i in range(2):
                            self.process_data(keep_pose)
                            for j in range(15):
                                self.env.step(keep_pose)
                        button_pos[:, 0] += self.env.gripper_length * 2 + 0.012
                        for i in range(2):
                            self.process_data(button_pos)
                            for j in range(15):
                                self.env.step(button_pos)
                        button_pos[:, 0] -= self.env.gripper_length
                        for i in range(2):
                            self.process_data(button_pos)
                            for j in range(15):
                                self.env.step(button_pos)
                        self.env.gripper = torch.ones((self.env.num_envs, 1), device=self.env.device)
                        self.process_data(button_pos)
                        for j in range(15):
                            self.env.step(button_pos)
                        button_pos[:, 0] -= 0.03
                        for i in range(2):
                            self.process_data(button_pos)
                            for j in range(15):
                                self.env.step(button_pos)
                        self.env.gripper = torch.zeros((self.env.num_envs, 1), device=self.env.device)

                        self.set_current_step(pull_step_idx, "拉门")
                        keep_pose = self.env.hand_rigid_body_tensor.clone()
                        keep_pose[:, 0] += self.env.gripper_length
                        self.process_data(keep_pose)
                        for j in range(15):
                            self.env.step(keep_pose)
                        handle_pos = pose[:, :7].clone()
                        handle_pos[:, 0] += self.env.gripper_length * 2
                        for i in range(2):
                            self.process_data(handle_pos)
                            for j in range(15):
                                self.env.step(handle_pos)
                        handle_pos[:, 0] -= self.env.gripper_length + 0.014
                        for i in range(2):
                            self.process_data(handle_pos)
                            for j in range(15):
                                self.env.step(handle_pos)
                        self.env.gripper = torch.ones((self.env.num_envs, 1), device=self.env.device)
                        self.process_data(handle_pos)
                        for j in range(15):
                            self.env.step(handle_pos)

                        down_q = torch.stack(self.env.num_envs * [torch.tensor([0.7071068, 0.7071068, 0, 0])]).to(self.env.device).view((self.env.num_envs, 4))
                        step_size = 0.045
                        for i in range(8):
                            handle_q = self.env.rigid_body_tensor[:, 3:7]
                            open_dir = quat_axis(handle_q, axis=2)
                            cur_p = self.env.hand_rigid_body_tensor[:, :3]
                            pred_p = cur_p + open_dir * step_size
                            pred_q = quat_mul(handle_q, down_q)
                            pred_pose = torch.cat([pred_p, pred_q], dim=-1).float()
                            self.process_data(pred_pose)
                            for j in range(15):
                                self.env.step(pred_pose)

                for env_id in range(self.env.num_envs):
                    if self.task_success_for_env(env_id):
                        demo_buffer.append(self.eps_buffer[env_id])
                        done_flag[env_id] = True
                        print(f"Env {env_id} Succeeded")
            finally:
                # ``collect_episode_end`` walks done_flag to build per-env
                # trajectory_records (driven by ``concrete_attempt_chain_for_collect``
                # for attempt_chain/stage_status, and ``ground_truth_chain_for_collect``
                # for the env-state-derived optimal). minimal_chain is
                # extracted from attempt_chain inside collect_episode_end.
                self.collect_episode_end(ctx, eps, done_flag)
        self.collect_finalize(ctx, demo_buffer)

    def action_choose(self, t, index, one_motion, two_motion):
        if "r" in self.env.action_chosen[index]:
            if one_motion > 0.0001:
                self.env.action_chosen[index, t] = "z"
                return "z"
            else:
                if two_motion > 0.05:
                    res = random.choice(["z", "o"])
                    self.env.action_chosen[index, t] = res
                    return res
                else:
                    if "z" == self.env.action_chosen[index, t - 1]:
                        self.env.action_chosen[index, t] = "o"
                        return "o"
                    else:
                        self.env.action_chosen[index, t] = "z"
                        return "z"
        else:
            if one_motion > 0.0001:
                self.env.action_chosen[index, t] = "z"
                return "z"
            else:
                if two_motion > 0.05:
                    res = random.choice(["z", "o"])
                    self.env.action_chosen[index, t] = res
                    return res
                else:
                    if "o" == self.env.action_chosen[index, t - 1]:
                        self.env.action_chosen[index, t] = "z"
                        return "z"
                    else:
                        if t == 0:
                            res = random.choice(["z", "o", "r"])
                            print("random")
                            self.env.action_chosen[index, t] = res
                            return res
                        elif "z" in self.env.action_chosen[index, t - 1]:
                            if "o" in self.env.action_chosen[index]:
                                self.env.action_chosen[index, t] = "o"
                                return "o"
                            else:
                                res = random.choice(["o", "r"])
                                self.env.action_chosen[index, t] = res
                                return res
