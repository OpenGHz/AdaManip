from manipulation.base_manipulation import BaseManipulation
from envs.base_env import BaseEnv
from manipulation.utils.transform import *
from logging import Logger
from typing import Any, Dict, List, Optional
import numpy as np
from dataset.dataset import Experience, Episode_Buffer, obs_wrapper
import os
import collections
# from utils.o3dviewer import torch2o3d

class OpenSafeManipulation(BaseManipulation) :

    # clock_wise=0 → chain 0 (拉门 — direct pull, no rotation needed)
    # clock_wise=1 → chain 1 (顺时针旋转旋钮 -> 拉门)
    # clock_wise=2 → chain 2 (逆时针旋转旋钮 -> 拉门)
    _CHAIN_DIRECT: List[str] = ["拉门"]
    _CHAIN_CW: List[str] = ["顺时针旋转旋钮", "拉门"]
    _CHAIN_CCW: List[str] = ["逆时针旋转旋钮", "拉门"]

    def __init__(self, env : BaseEnv, cfg : dict, logger : Logger) :

        super().__init__(env, cfg, logger)

    # ------------------------------------------------------------------
    # Hooks consumed by BaseManipulation.diffusion_evaluate
    # ------------------------------------------------------------------

    def language_template_task_name(self) -> str:
        return "safe"

    def dataset_dir_suffix(self) -> str:
        return "clock" + str(self.cfg["env"]["clockwise"])

    def reset_kwargs_initial(self) -> Dict[str, Any]:
        return {"clock_same": False}

    def task_success_for_env(self, env_id: int) -> bool:
        return bool(
            (torch.abs(self.env.one_dof_tensor[env_id, 0]) > np.pi / 7).cpu().item()
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
        cw_int = int(round(float(cw)))
        if cw_int == 0:
            return list(self._CHAIN_DIRECT)
        if cw_int == 1:
            return list(self._CHAIN_CW)
        if cw_int == 2:
            return list(self._CHAIN_CCW)
        return None

    def per_env_extra_log_fields(
        self, env_id: int, episode_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        clock_wise = episode_state.get("clock_wise") if episode_state else None
        return {
            "clock_wise": float(clock_wise) if clock_wise is not None else None,
            "final_one_dof": float(self.env.one_dof_tensor[env_id, 0].item()),
        }

    def concrete_attempt_chain_for_collect(self, env_id: int, state: Dict[str, Any]):
        # Reconstruct attempt_chain from per-env frame records (populated
        # via set_current_step + process_data → append_frame_label). Each
        # contiguous run of identical step_operation collapses into one
        # stage; status is derived from cw (rotate matches cw → True;
        # intermediate pull → False; final stage → True).
        records = self._episode_frame_records
        if records is None or env_id >= len(records) or not records[env_id]:
            return super().concrete_attempt_chain_for_collect(env_id, state)
        attempt = []
        last_op = None
        for fr in records[env_id]:
            op = fr.get("step_operation")
            if not op:
                continue
            if op != last_op:
                attempt.append(op)
                last_op = op
        if not attempt:
            return super().concrete_attempt_chain_for_collect(env_id, state)
        cw = int(round(float(state.get("clock_wise", 0)))) if state else None
        statuses = []
        for i, op in enumerate(attempt):
            is_last = (i == len(attempt) - 1)
            if is_last:
                statuses.append(True)
            elif op == "拉门":
                statuses.append(False)
            elif op == "顺时针旋转旋钮":
                statuses.append(cw == 1)
            elif op == "逆时针旋转旋钮":
                statuses.append(cw == 2)
            else:
                statuses.append(False)
        return attempt, statuses

    '''
    test env
    '''
    def test_env(self, pose, eval=False):
        batch_size = pose.shape[0]
        handle_pos = pose[:,:7]
        knob_pos = pose[:,7:]
        '''
        two manipulation choice 1.pull handle open door 2.push knob then pull handle open door
        '''
        self.env.reset()
        flag = False
        if flag:
            handle_pos[:, 0] += self.env.gripper_length*2
            for i in range(3):
                for j in range(15):
                    self.env.step(handle_pos)
            handle_pos[:, 0] -= self.env.gripper_length
            for i in range(2):
                for j in range(15):
                    self.env.step(handle_pos)
            self.env.gripper = True
            for i in range(1):
                for j in range(15):
                    self.env.step(handle_pos)
            down_q = torch.stack(self.env.num_envs * [torch.tensor([0, 1, 0, 0])]).to(self.env.device).view((self.env.num_envs, 4))
            step_size = 0.04
            for i in range(10):
                print("step_{}".format(i))
                handle_q = self.env.handle_rigid_body_tensor[:, 3:7]
                open_dir = quat_axis(handle_q, axis=2)
                cur_p = self.env.hand_rigid_body_tensor[:, :3]
                pred_p = cur_p + open_dir * step_size
                pred_q = quat_mul(handle_q, down_q)
                pred_pose = torch.cat([pred_p, pred_q], dim=-1).float()
                for j in range(15):
                    self.env.step(pred_pose)
        else:
            hand_pose = self.env.hand_rigid_body_tensor[:,:7]
            for i in range(10000000):
                self.env.step(hand_pose)
            init_handle_pos = handle_pos.clone()
            init_handle_pos[:, 0] += self.env.gripper_length*2
            for i in range(4):
                for j in range(15):
                    self.env.step(init_handle_pos)
                    
            init_handle_pos[:, 0] -= self.env.gripper_length
            for i in range(3):
                for j in range(15):
                    self.env.step(init_handle_pos)
            self.env.gripper = True
            for i in range(2):
                for j in range(15):
                    self.env.step(init_handle_pos)

            down_q = torch.stack(self.env.num_envs * [torch.tensor([0, 1, 0, 0])]).to(self.env.device).view((self.env.num_envs, 4))
            step_size = 0.04
            for i in range(2):
                handle_q = self.env.rigid_body_tensor[:, 3:7]
                open_dir = quat_axis(handle_q, axis=2)
                cur_p = self.env.hand_rigid_body_tensor[:, :3]
                pred_p = cur_p + open_dir * step_size
                pred_q = quat_mul(handle_q, down_q)
                pred_pose = torch.cat([pred_p, pred_q], dim=-1).float()
                for j in range(15):
                    self.env.step(pred_pose)
            self.env.gripper = False
            for j in range(15):
                self.env.step(pred_pose)
            knob_pos[:, 0] += self.env.gripper_length*2
            for i in range(3):
                for j in range(15):
                    self.env.step(knob_pos)
            knob_pos[:, 0] -= self.env.gripper_length
            for i in range(3):
                for j in range(15):
                    self.env.step(knob_pos)
            self.env.gripper = True
            for i in range(1):
                for j in range(15):
                    self.env.step(knob_pos)
            rot_quat = torch.tensor([[ 0, 0, 0.1305262, 0.9914449]]*batch_size, device=self.env.device) 
            s_rot_quat = torch.tensor([[ 0, 0, -0.1305262, 0.9914449]]*batch_size, device=self.env.device) 

            for i in range(2):
                handle_q = self.env.part_rigid_body_tensor[:, 3:7]
                cur_p = self.env.hand_rigid_body_tensor[:, :3]
                cur_q = self.env.hand_rigid_body_tensor[:,3:7]
                pred_p = cur_p
                pred_q = quat_mul(cur_q, s_rot_quat)
                pred_pose = torch.cat([pred_p, pred_q], dim=-1).float()
                for j in range(15):
                    self.env.step(pred_pose)

            for i in range(8):
                handle_q = self.env.part_rigid_body_tensor[:, 3:7]
                cur_p = self.env.hand_rigid_body_tensor[:, :3]
                cur_q = self.env.hand_rigid_body_tensor[:,3:7]
                pred_p = cur_p
                pred_q = quat_mul(cur_q, rot_quat)
                pred_pose = torch.cat([pred_p, pred_q], dim=-1).float()
                for j in range(15):
                    self.env.step(pred_pose)

            self.env.gripper = False
            for i in range(1):
                for j in range(15):
                    self.env.step(pred_pose)

            handle_pos[:, 0] += self.env.gripper_length*2
            for i in range(3):
                for j in range(15):
                    self.env.step(handle_pos)
            handle_pos[:, 0] -= self.env.gripper_length
            for i in range(2):
                for j in range(15):
                    self.env.step(handle_pos)

            self.env.gripper = True
            for i in range(1):
                for j in range(15):
                    self.env.step(handle_pos)

            down_q = torch.stack(self.env.num_envs * [torch.tensor([0, 1, 0, 0])]).to(self.env.device).view((self.env.num_envs, 4))
            step_size = 0.04
            for i in range(10):
                print("step_{}".format(i))
                handle_q = self.env.handle_rigid_body_tensor[:, 3:7]
                open_dir = quat_axis(handle_q, axis=2)
                cur_p = self.env.hand_rigid_body_tensor[:, :3]
                pred_p = cur_p + open_dir * step_size
                pred_q = quat_mul(handle_q, down_q)
                pred_pose = torch.cat([pred_p, pred_q], dim=-1).float()
                for j in range(15):
                    self.env.step(pred_pose)
    
    # diffusion_evaluate is provided by BaseManipulation. Set
    # task.max_step in cfg to control inner-loop bound (this task historically
    # used 50; default base value is 32).


    def process_data(self, goal_pos):
        obs = self.env.collect_diff_data()
        pc, env_state = obs_wrapper(obs)
        goal_pos = self.action_process(goal_pos)
        if self.env.gripper[0,0].cpu().item() == 1:
            temp = torch.ones((self.env.num_envs,1),device=self.env.device)
        else:
            temp = torch.zeros((self.env.num_envs,1),device=self.env.device)
        action_with_gripper = torch.cat([goal_pos, temp],dim=-1)
        self.env.actions = action_with_gripper
        for env_id in range(self.env.num_envs):
            self.all_eps_buffer[env_id].add(pc[env_id], env_state[env_id],action_with_gripper[env_id])
            self.append_frame_label(env_id)
        self._record_video_frame()


    def collect_manip_data(self):
        eps_num = self.cfg["task"]["num_episode"]
        policy = self.cfg["task"]["policy"]
        rot_quat = torch.tensor([ 0, 0, -0.258819, 0.9659258], device=self.env.device)
        s_rot_quat = torch.tensor([ 0, 0, 0.258819, 0.9659258], device=self.env.device)

        # Unified-type task: no role suffix → dataset dir is `open_safe_<policy>_*`.
        ctx = self.collect_setup(role=None)
        all_demo_buffer = Experience() # Save the continuous action trajectory in the whole episode
        for eps in range(eps_num):
            self.all_eps_buffer = [Episode_Buffer() for _ in range(self.env.num_envs)]
            print("eps_{}".format(eps+1))
            self.env.reset()
            self.collect_episode_start(ctx, eps)
            # print(self.env.clock_wise)
            pose = self.env.get_adjust_hand_pose().clone()
            handle_pos = pose[:,:7]
            knob_pos = pose[:,7:]
            knob_pos[:,0] -= 0.002
            if policy == "succ":
                if self.env.clock_wise[0]: # locked
                    cw_root = int(self.env.clock_wise[0].item())
                    rotate_op = "顺时针旋转旋钮" if cw_root == 1 else "逆时针旋转旋钮"
                    # All approach / grasp / rotate-knob frames belong to the
                    # rotate-knob stage (step_index 0).
                    self.set_current_step(0, rotate_op)
                    self.env.gripper = torch.zeros((self.env.num_envs,1), device=self.env.device)
                    for i in range(2):
                        self.process_data(self.env.hand_rigid_body_tensor[:,:7])
                        for j in range(15):
                            self.env.step(self.env.hand_rigid_body_tensor[:,:7])
                    # grasp knob

                    knob_pos[:, 0] += self.env.gripper_length*2
                    for i in range(2):
                        self.process_data(knob_pos)
                        for j in range(15):
                            self.env.step(knob_pos)

                    knob_pos[:, 0] -= self.env.gripper_length
                    for i in range(2):
                        self.process_data(knob_pos)
                        for j in range(15):
                            self.env.step(knob_pos)

                    self.env.gripper = torch.ones((self.env.num_envs,1), device=self.env.device)
                    for i in range(2):
                        self.process_data(knob_pos)
                        for j in range(15):
                            self.env.step(knob_pos)

                    # rotate knob
                    for i in range(5):
                        cur_p = self.env.hand_rigid_body_tensor[:, :3]
                        cur_q = self.env.hand_rigid_body_tensor[:,3:7]
                        pred_p = cur_p
                        for i in range(self.env.num_envs):
                            if self.env.clock_wise[i] == 1:
                                # counter clock wise
                                cur_q[i] = quat_mul(cur_q[i], rot_quat)
                            else:
                                # clock wise == 2
                                cur_q[i] = quat_mul(cur_q[i], s_rot_quat)
                        pred_pose = torch.cat([pred_p, cur_q], dim=-1).float()

                        self.process_data(pred_pose)
                        for j in range(15):
                            self.env.step(pred_pose)

                    # Transition to the pull-door stage (step_index 1).
                    self.set_current_step(1, "拉门")
                    self.env.gripper = torch.zeros((self.env.num_envs,1), device=self.env.device)
                    for i in range(2):
                        self.process_data(self.env.hand_rigid_body_tensor[:,:7])
                        for j in range(15):
                            self.env.step(self.env.hand_rigid_body_tensor[:,:7])
                    # grasp handle

                    handle_pos[:, 0] += self.env.gripper_length*2
                    for i in range(3):
                        self.process_data(handle_pos)
                        for j in range(15):
                            self.env.step(handle_pos)

                    handle_pos[:, 0] -= self.env.gripper_length
                    for i in range(2):
                        self.process_data(handle_pos)
                        for j in range(15):
                            self.env.step(handle_pos)

                    self.env.gripper = torch.ones((self.env.num_envs,1), device=self.env.device)
                    for i in range(2):
                        self.process_data(handle_pos)
                        for j in range(15):
                            self.env.step(handle_pos)

                    down_q = torch.stack(self.env.num_envs * [torch.tensor([0, 1, 0, 0])]).to(self.env.device).view((self.env.num_envs, 4))
                    step_size = 0.04
                    for i in range(10):
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
                    # cw=0 (unlocked): direct pull-door, single stage.
                    self.set_current_step(0, "拉门")
                    self.env.gripper = torch.zeros((self.env.num_envs,1), device=self.env.device)
                    for i in range(2):
                        self.process_data(self.env.hand_rigid_body_tensor[:,:7])
                        for j in range(15):
                            self.env.step(self.env.hand_rigid_body_tensor[:,:7])

                    init_handle_pos = handle_pos.clone()
                    init_handle_pos[:, 0] += self.env.gripper_length*2
                    for i in range(4):
                        self.process_data(init_handle_pos)
                        for j in range(15):
                            self.env.step(init_handle_pos)

                    init_handle_pos[:, 0] -= self.env.gripper_length
                    for i in range(3):
                        self.process_data(init_handle_pos)
                        for j in range(15):
                            self.env.step(init_handle_pos)

                    self.env.gripper = torch.ones((self.env.num_envs,1), device=self.env.device)
                    for i in range(2):
                        self.process_data(init_handle_pos)
                        for j in range(15):
                            self.env.step(init_handle_pos)
                    
                    # open door
                    down_q = torch.stack(self.env.num_envs * [torch.tensor([0, 1, 0, 0])]).to(self.env.device).view((self.env.num_envs, 4))
                    step_size = 0.04
                    for i in range(10):
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
                    if (torch.abs(self.env.one_dof_tensor[env_id, 0]) > np.pi/6).cpu().item():
                        all_demo_buffer.append(self.all_eps_buffer[env_id])
                        print(f"Env {env_id} Succeeded")

            else:
                # adaptive demo: starts by attempting pull (whether
                # start_with_pull is True or False, the demo first goes
                # for the handle). Default phase = pull.
                self.set_current_step(0, "拉门")
                self.env.gripper = torch.zeros((self.env.num_envs,1), device=self.env.device)
                for i in range(2):
                    self.process_data(self.env.hand_rigid_body_tensor[:,:7])
                    for j in range(15):
                        self.env.step(self.env.hand_rigid_body_tensor[:,:7])

                # grasp handle
                init_handle_pos = handle_pos.clone()
                init_handle_pos[:, 0] += self.env.gripper_length*2
                for i in range(4):
                    self.process_data(init_handle_pos)
                    for j in range(15):
                        self.env.step(init_handle_pos)

                # move to handle
                init_handle_pos[:, 0] -= self.env.gripper_length
                for i in range(3):
                    self.process_data(init_handle_pos)
                    for j in range(15):
                        self.env.step(init_handle_pos)

                # close gripper
                self.env.gripper = torch.ones((self.env.num_envs,1), device=self.env.device)
                for i in range(2):
                    self.process_data(init_handle_pos)
                    for j in range(15):
                        self.env.step(init_handle_pos)

                # open door
                down_q = torch.stack(self.env.num_envs * [torch.tensor([0, 1, 0, 0])]).to(self.env.device).view((self.env.num_envs, 4))
                step_size = 0.04
                start_with_pull = np.random.rand() < 0.5

                if start_with_pull:
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

                if start_with_pull and not self.env.clock_wise[0]:
                    # open door directly
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
                    # First pull attempt failed (locked) — fall back to
                    # rotate-knob. attempt_chain becomes
                    # [拉门(failed), {direction}旋转旋钮, 拉门].
                    cw_root = int(self.env.clock_wise[0].item())
                    rotate_op = "顺时针旋转旋钮" if cw_root == 1 else "逆时针旋转旋钮"
                    self.set_current_step(1, rotate_op)
                    self.env.gripper = torch.zeros((self.env.num_envs,1), device=self.env.device)

                    self.open_eps_buffer = [Episode_Buffer() for _ in range(self.env.num_envs)]

                    for i in range(2):
                        self.process_data(self.env.hand_rigid_body_tensor[:,:7])
                        for j in range(15):
                            self.env.step(self.env.hand_rigid_body_tensor[:,:7])

                    # grasp knob
                    knob_pos[:, 0] += self.env.gripper_length*2
                    for i in range(2):
                        self.process_data(knob_pos)
                        for j in range(15):
                            self.env.step(knob_pos)

                    knob_pos[:, 0] -= self.env.gripper_length
                    for i in range(2):
                        self.process_data(knob_pos)
                        for j in range(15):
                            self.env.step(knob_pos)

                    self.env.gripper = torch.ones((self.env.num_envs,1), device=self.env.device)
                    for i in range(2):
                        self.process_data(knob_pos)
                        for j in range(15):
                            self.env.step(knob_pos)

                    # rotate knob
                    for t in range(6):
                        cur_p = self.env.hand_rigid_body_tensor[:, :3]
                        cur_q = self.env.hand_rigid_body_tensor[:,3:7]
                        pred_p = cur_p
                        for i in range(self.env.num_envs):
                            if t == 0:
                                if np.random.rand() > 0.5:
                                    cur_q[i] = quat_mul(cur_q[i], rot_quat)
                                else:
                                    cur_q[i] = quat_mul(cur_q[i], s_rot_quat)
                            else:
                                if self.env.clock_wise[i] == 1:
                                    cur_q[i] = quat_mul(cur_q[i], rot_quat)
                                else:
                                    cur_q[i] = quat_mul(cur_q[i], s_rot_quat)
                        pred_pose = torch.cat([pred_p, cur_q], dim=-1).float()
                        self.process_data(pred_pose)
                        for j in range(15):
                            self.env.step(pred_pose)
                    
                    # After rotate-knob, transition back to pull-door
                    # (final stage 2 in this 3-stage attempt).
                    self.set_current_step(2, "拉门")
                    self.env.gripper = torch.zeros((self.env.num_envs,1), device=self.env.device)
                    for i in range(2):
                        self.process_data(self.env.hand_rigid_body_tensor[:,:7])
                        for j in range(15):
                            self.env.step(self.env.hand_rigid_body_tensor[:,:7])

                    # grasp handle
                    handle_pos[:, 0] += self.env.gripper_length*2
                    for i in range(3):
                        self.process_data(handle_pos)
                        for j in range(15):
                            self.env.step(handle_pos)

                    handle_pos[:, 0] -= self.env.gripper_length
                    for i in range(2):
                        self.process_data(handle_pos)
                        for j in range(15):
                            self.env.step(handle_pos)

                    self.env.gripper = torch.ones((self.env.num_envs,1), device=self.env.device)
                    for i in range(2):
                        self.process_data(handle_pos)
                        for j in range(15):
                            self.env.step(handle_pos)

                    # open door
                    for i in range(10):
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
                    all_demo_buffer.append(self.all_eps_buffer[env_id])
                    print(f"Env {env_id} Succeeded")
            done_flag = [
                bool((torch.abs(self.env.one_dof_tensor[env_id, 0]) > np.pi / 6).cpu().item())
                for env_id in range(self.env.num_envs)
            ]
            self.collect_episode_end(ctx, eps, done_flag)

        self.collect_finalize(ctx, all_demo_buffer)


