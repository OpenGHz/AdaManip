from manipulation.base_manipulation import BaseManipulation
from envs.base_env import BaseEnv
from manipulation.utils.transform import *
from logging import Logger
from typing import Any, Dict, List, Optional
import numpy as np
from dataset.dataset import Experience, Episode_Buffer, obs_wrapper
import os
import collections

map_ = {'z':1, 'y':2, 'r':3}

class OpenLampManipulation(BaseManipulation) :

    # clock_wise=1 → chain 0 (推开关 — push switch)
    # clock_wise=2 → chain 2 (逆时针旋转开关 — env comment says "counter clock wise")
    # clock_wise=3 → chain 1 (顺时针旋转开关)
    _CHAIN_PUSH: List[str] = ["推开关"]
    _CHAIN_CW: List[str] = ["顺时针旋转开关"]
    _CHAIN_CCW: List[str] = ["逆时针旋转开关"]

    def __init__(self, env : BaseEnv, cfg : dict, logger : Logger) :

        super().__init__(env, cfg, logger)

    # ------------------------------------------------------------------
    # Hooks consumed by BaseManipulation.diffusion_evaluate
    # ------------------------------------------------------------------

    def language_template_task_name(self) -> str:
        return "lamp"

    def dataset_dir_suffix(self) -> str:
        return "clock" + str(self.cfg["env"]["clockwise"])

    def task_success_for_env(self, env_id: int) -> bool:
        # Lamp's success criterion depends on the per-env mode:
        # clock_wise == 1 → push switch (translation) → check one_dof > 0.01
        # clock_wise != 1 → rotate switch → check two_dof past two_flag
        if int(self.env.clock_wise[env_id].item()) == 1:
            return bool(
                (torch.abs(self.env.one_dof_tensor[env_id, 0]) > 0.01).cpu().item()
            )
        return bool(
            (
                torch.abs(self.env.two_dof_tensor[env_id, 0])
                > torch.abs(self.env.two_flag[env_id])
            ).cpu().item()
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
        if cw_int == 1:
            return list(self._CHAIN_PUSH)
        if cw_int == 3:
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
            "final_two_dof": float(self.env.two_dof_tensor[env_id, 0].item()),
        }

    def concrete_attempt_chain_for_collect(self, env_id: int, state: Dict[str, Any]):
        # Lamp's ada_policy randomly picks one of {push, cw rotate, ccw
        # rotate} at t=0 and switches if the first pick wasn't cw-correct.
        # attempt_chain captures those retries.
        chains = getattr(self, "_lamp_attempt_chains", None)
        statuses = getattr(self, "_lamp_stage_statuses", None)
        if (
            chains is not None and statuses is not None
            and env_id < len(chains) and chains[env_id]
        ):
            return list(chains[env_id]), list(statuses[env_id])
        return super().concrete_attempt_chain_for_collect(env_id, state)
    '''
    test env
    '''
    def test_env(self, pose, eval=False):
        batch_size = pose.shape[0]
        self.env.reset()
        pose[:, 2] += self.env.gripper_length*2
        for i in range(2):
            for j in range(15):
                self.env.step(pose)
        pose[:, 2] -= self.env.gripper_length + 0.01
        for i in range(2):
            for j in range(15):
                self.env.step(pose)
        self.env.gripper = True
        for i in range(1000000):
            for j in range(15):
                self.env.step(pose)
        
        '''
        two choice
        '''
        rot_quat = torch.tensor([[ 0, 0, 0.1305262, 0.9914449]]*batch_size, device=self.env.device) 
        flag = True
        step_size = 0.01
        if flag:# rotate
            for i in range(10):
                cur_p = self.env.hand_rigid_body_tensor[:, :3]
                cur_q = self.env.hand_rigid_body_tensor[:,3:7]
                pred_p = cur_p
                pred_q = quat_mul(cur_q, rot_quat)
                pred_pose = torch.cat([pred_p, pred_q], dim=-1).float()
                for j in range(15):
                    self.env.step(pred_pose)   
        else:
            for i in range(4):
                print(self.env.one_dof_tensor[:,0])
                cur_p = self.env.hand_rigid_body_tensor[:, :3]
                cur_q = self.env.hand_rigid_body_tensor[:,3:7]
                cur_p[:, 2] = cur_p[:,2] - step_size
                pred_pose = torch.cat([cur_p, cur_q], dim=-1).float()
                for j in range(15):
                    self.env.step(pred_pose)    
       
    '''
    model test
    '''
    # diffusion_evaluate is provided by BaseManipulation.



    '''
    eval grasp net
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
            pre_pose[:, 2] -= self.env.gripper_length + 0.01
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
        primitive_action_step = 0.01
        rot_quat = torch.tensor([ 0, 0, -0.1305262, 0.9914449], device=self.env.device)
        s_rot_quat = torch.tensor([ 0, 0, 0.1305262, 0.9914449], device=self.env.device)
        ctx = self.collect_setup(role="manip")
        demo_buffer = Experience()
        hand_pose = self.env.hand_rigid_body_tensor[:,:7]
        max_step = 15 if policy == "adaptive" else 10
        succ_cnt = [0] * self.env.num_envs
        for eps in range(eps_num):
            chose_list = [['z','y','r'] for _ in range(self.env.num_envs)]
            eps_buffer = [Episode_Buffer() for _ in range(self.env.num_envs)]
            done_flag = [False] * self.env.num_envs
            print("eps_{}".format(eps+1))
            self.env.reset()
            self.collect_episode_start(ctx, eps)
            pre_pose = self.env.adjust_hand_pose.clone()
            pre_pose[:, 2] += self.env.gripper_length*2
            for i in range(3):
                for j in range(10):
                    self.env.step(pre_pose)
                self._record_video_frame()

            pre_pose[:, 2] -= self.env.gripper_length + 0.01
            for i in range(3):
                for j in range(10):
                    self.env.step(pre_pose)
                self._record_video_frame()

            self.env.gripper = True
            for i in range(10):
                self.env.step(hand_pose)
            self._record_video_frame()

            init_actions = self.action_process(hand_pose)
            self.env.actions = init_actions
            ####################start collect manipulation data###################
            prev_op_for_env = [None] * self.env.num_envs
            step_idx_for_env = [0] * self.env.num_envs
            res_to_op = {
                "z": "推开关",
                "r": "顺时针旋转开关",
                "y": "逆时针旋转开关",
            }
            self._lamp_attempt_chains = [[] for _ in range(self.env.num_envs)]
            self._lamp_stage_statuses = [[] for _ in range(self.env.num_envs)]
            current_op = [None] * self.env.num_envs

            def _stage_status_for_intermediate(env_id, op):
                # Lamp success criterion is op-vs-cw match:
                #   cw=1 → 推开关 / cw=2 → 逆时针旋转 / cw=3 → 顺时针旋转
                cw = int(self.env.clock_wise[env_id].item())
                if op == "推开关":
                    return cw == 1
                if op == "逆时针旋转开关":
                    return cw == 2
                if op == "顺时针旋转开关":
                    return cw == 3
                return False

            def _flush_stage(env_id, op, success):
                self._lamp_attempt_chains[env_id].append(op)
                self._lamp_stage_statuses[env_id].append(bool(success))

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
                        res = self.ada_policy(i, t, chose_list[i])
                    else:
                        raise NotImplementedError
                    res_per_env.append(res)
                    if res == "z":
                        pre_p[i,2] -= primitive_action_step
                    elif res == "r":
                        pre_q[i] = quat_mul(cur_q[i], s_rot_quat)
                    elif res == "y":
                        pre_q[i] = quat_mul(cur_q[i], rot_quat)

                pred_pose = torch.cat([pre_p, pre_q], dim=-1).float()
                gt_pose = self.action_process(pred_pose)

                for env_id in range(self.env.num_envs):
                    if not done_flag[env_id]:
                        obs = self.env.collect_single_diff_data(env_id)
                        pc, env_state = obs_wrapper(obs)
                        eps_buffer[env_id].add(pc, env_state, gt_pose[env_id])
                        op = res_to_op.get(res_per_env[env_id])
                        if op is not None:
                            if prev_op_for_env[env_id] is None:
                                current_op[env_id] = op
                            elif prev_op_for_env[env_id] != op:
                                _flush_stage(
                                    env_id,
                                    current_op[env_id],
                                    _stage_status_for_intermediate(env_id, current_op[env_id]),
                                )
                                step_idx_for_env[env_id] += 1
                                current_op[env_id] = op
                            prev_op_for_env[env_id] = op
                            self.append_frame_label_for(env_id, step_idx_for_env[env_id], op)

                for j in range(15):
                    self.env.step(pred_pose)
                self._record_video_frame()

                self.env.actions = gt_pose

                # update env end flag
                for env_id in range(self.env.num_envs):
                    if not done_flag[env_id]:
                        if self.env.clock_wise[env_id] == 1:
                            if torch.abs(self.env.one_dof_tensor[env_id, 0]) > 0.007:
                                if current_op[env_id] is not None:
                                    _flush_stage(env_id, current_op[env_id], True)
                                    current_op[env_id] = None
                                demo_buffer.append(eps_buffer[env_id])
                                done_flag[env_id] = True
                                succ_cnt[env_id] += 1
                                print(f"Env {env_id} Succeeded")
                        else:
                            if torch.abs(self.env.two_dof_tensor[env_id, 0]) > torch.abs(self.env.two_flag[env_id]):
                                if current_op[env_id] is not None:
                                    _flush_stage(env_id, current_op[env_id], True)
                                    current_op[env_id] = None
                                demo_buffer.append(eps_buffer[env_id])
                                done_flag[env_id] = True
                                succ_cnt[env_id] += 1
                                print(f"Env {env_id} Succeeded")
            print(succ_cnt)
            self.collect_episode_end(ctx, eps, done_flag)

        self.collect_finalize(ctx, demo_buffer)
    
    def succ_policy(self, env_id):
        clock_wise = self.env.clock_wise[env_id]
        if clock_wise == 1:
            return 'z'
        elif clock_wise == 2:
            return 'y'
        elif clock_wise == 3:
            return 'r'
    
    def ada_policy(self, env_id, t, cho_list):
        clock_wise = self.env.clock_wise[env_id]
        prob = np.random.rand()*3
        if t == 0:
            if prob < 1:
                self.env.action_chosen[env_id, t] = 'z'
                cho_list.remove('z')
                return 'z'
            elif prob < 2:
                self.env.action_chosen[env_id, t] = 'y'
                cho_list.remove('y')
                return 'y'
            else:
                self.env.action_chosen[env_id, t] = 'r'
                cho_list.remove('r')
                return 'r'
        
        if map_[self.env.action_chosen[env_id, t - 1]] == clock_wise:
            action = self.env.action_chosen[env_id, t - 1]
            self.env.action_chosen[env_id, t] = action
            return action
        else:
            if len(cho_list) == 2:
                s_prob = np.random.rand()
                if s_prob < 11/20:
                    action = cho_list[0]
                    cho_list.remove(action)
                    self.env.action_chosen[env_id, t] = action
                    return action
                else:
                    action = cho_list[1]
                    cho_list.remove(action)
                    self.env.action_chosen[env_id, t] = action
                    return action
            else:
                action = cho_list[0]
                cho_list.remove(action)
                self.env.action_chosen[env_id, t] = action
                return action