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
    # ``_build_microwave_trajectory_label`` demo logic.
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

    # ------------------------------------------------------------------
    # Microwave demonstration helpers
    # ------------------------------------------------------------------

    def _build_microwave_trajectory_label(self, env_id, start_with_pull, expanded_minimal_chains):
        clock_wise = int(self.env.clock_wise[env_id].item())
        if start_with_pull:
            if clock_wise == 0:
                attempt_chain = list(self._UNLOCKED_CHAIN)
                stage_status = [True]
                minimal_chain = list(self._UNLOCKED_CHAIN)
            else:
                attempt_chain = ["拉门", "按按钮", "拉门"]
                stage_status = [False, True, True]
                minimal_chain = list(self._LOCKED_CHAIN)
        else:
            attempt_chain = list(self._LOCKED_CHAIN)
            stage_status = [True, True]
            minimal_chain = list(self._LOCKED_CHAIN)

        try:
            minimal_chain_id = expanded_minimal_chains.index(minimal_chain)
        except ValueError as exc:
            raise RuntimeError(
                f"minimal_chain {minimal_chain} not found in expanded_minimal_chains"
            ) from exc

        command_chains, command_chain_ids = self.match_command_chains(
            attempt_chain=attempt_chain,
            stage_status=stage_status,
            expanded_minimal_chains=expanded_minimal_chains,
        )

        return {
            "minimal_chain_id": minimal_chain_id,
            "minimal_chain": minimal_chain,
            "attempt_chain": attempt_chain,
            "stage_status": stage_status,
            "command_chains": command_chains,
            "command_chain_ids": command_chain_ids,
            "success": True,
        }

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
        template_path, task_spec = self.load_task_language_template(
            self.language_template_task_name()
        )
        expanded_minimal_chains = self.build_expanded_minimal_chains(task_spec["minimal_chains"])

        demo_buffer = Experience()
        trajectory_records = []
        frame_records = []
        saved_episode_id = 0

        save_dir = self._build_collect_save_dir()
        self._prepare_save_dir(save_dir, "Collection")
        self._init_video_recorder(save_dir)
        for eps in range(eps_num):
            self.eps_buffer = [Episode_Buffer() for _ in range(self.env.num_envs)]
            self.init_episode_frame_records(self.env.num_envs)
            print("eps_{}".format(eps + 1))
            self.env.reset()
            if self.video_recorder is not None:
                self.video_recorder.start_episode(eps)
            ori_pose = self.env.get_adjust_hand_pose()
            pose = ori_pose.clone()
            handle_pos = pose[:, :7].clone()
            button_pos = pose[:, 7:].clone()
            self.env.gripper = torch.zeros((self.env.num_envs, 1), device=self.env.device)
            episode_start_with_pull = False
            try:
                if policy == "succ":
                    if self.env.clock_wise[0] == 1:
                        episode_start_with_pull = False
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
                        episode_start_with_pull = True
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
                    episode_start_with_pull = start_with_pull

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
                        traj_record = self._build_microwave_trajectory_label(
                            env_id=env_id,
                            start_with_pull=episode_start_with_pull,
                            expanded_minimal_chains=expanded_minimal_chains,
                        )
                        traj_record["episode_id"] = saved_episode_id
                        traj_record["round_idx"] = eps
                        traj_record["env_id"] = env_id
                        frame_start = len(frame_records)
                        env_frames = self._episode_frame_records[env_id]
                        frame_records.extend(env_frames)
                        frame_end = len(frame_records)
                        traj_record["frame_range"] = [frame_start, frame_end]
                        trajectory_records.append(traj_record)
                        saved_episode_id += 1
                        print(f"Env {env_id} Succeeded")
            finally:
                if self.video_recorder is not None:
                    self.video_recorder.finish_episode()
                self.clear_episode_frame_records()

        if self.cfg["env"]["collectData"]:
            import os
            save_path = save_dir + "/demo_data.zip"
            os.makedirs(save_dir, exist_ok=True)
            demo_buffer.save(save_path)
            self.save_language_sidecars(
                save_dir=save_dir,
                template_path=template_path,
                task_name=self.language_template_task_name(),
                task_spec=task_spec,
                expanded_minimal_chains=expanded_minimal_chains,
                trajectory_records=trajectory_records,
                frame_records=frame_records,
            )
        self.video_recorder = None

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
