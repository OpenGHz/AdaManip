from manipulation.base_manipulation import BaseManipulation
from envs.base_env import BaseEnv
from manipulation.utils.transform import *
from logging import Logger
from typing import Any, Dict, List, Optional
import numpy as np
from dataset.dataset import Experience, Episode_Buffer, obs_wrapper
import os
import collections

class OpenCoffeeMachineManipulation(BaseManipulation) :

    def __init__(self, env : BaseEnv, cfg : dict, logger : Logger) :

        super().__init__(env, cfg, logger)

    # ------------------------------------------------------------------
    # Hooks consumed by BaseManipulation.diffusion_evaluate
    # ------------------------------------------------------------------

    def language_template_task_name(self) -> str:
        return "coffee_maker"

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
        # move to the handle
        pose[:, 2] += self.env.gripper_length*2
        self.env.reset()
        for i in range(30):
            self.env.step(pose)
        pose[:, 2] -= self.env.gripper_length + 0.008
        for i in range(30):
            self.env.step(pose)
        self.env.gripper = True
        for i in range(10):
            self.env.step(pose)
        down_q = torch.stack(self.env.num_envs * [torch.tensor([0, -0.7071068, 0.7071068, 0])]).to(self.env.device).view((self.env.num_envs, 4))
        # 0.7071068, 0.0, 0, 0.7071068
        step_size = 0.03
        
        for i in range(13):
            print("step_{}".format(i))
            handle_q = self.env.part_rigid_body_tensor[:, 3:7]
            
            open_dir = quat_axis(handle_q, axis=1)
            rotate_dir = quat_axis(handle_q, axis=0)

            cur_p = self.env.hand_rigid_body_tensor[:, :3]
            print(self.env.open_bottle_stage)
            pred_p = torch.where(self.env.open_bottle_stage.unsqueeze(1).repeat_interleave(3, dim=-1), 
                                 cur_p - open_dir * step_size, cur_p - rotate_dir*step_size)
            pred_q = quat_mul(handle_q, down_q)
            pred_pose = torch.cat([pred_p, pred_q], dim=-1).float()
            for j in range(15):
                self.env.step(pred_pose)   

    '''
    grasp net data collect
    '''
    def collect_grasp_data(self):
        eps_num = self.cfg["task"]["num_episode"]
        demo_buffer = Experience()
        for eps in range(eps_num):
            self.eps_buffer = [Episode_Buffer() for _ in range(self.env.num_envs)]
            print("eps_{}".format(eps+1))
            self.env.reset()
            pre_pose = self.env.adjust_hand_pose.clone()
            pre_pose[:, 2] += self.env.gripper_length*2
            for i in range(3):
                obs = self.env.collect_diff_data()
                pc, env_state = obs_wrapper(obs)

                for j in range(10):
                    self.env.step(pre_pose)
                
                gt_action = self.action_process(pre_pose)
                self.env.actions = gt_action.clone()
                for env_id in range(self.env.num_envs):
                    self.eps_buffer[env_id].add(pc[env_id], env_state[env_id], gt_action[env_id])
            pre_pose[:, 2] -= self.env.gripper_length + 0.008
            for i in range(3):
                obs = self.env.collect_diff_data()
                pc, env_state = obs_wrapper(obs)

                for j in range(10):
                    self.env.step(pre_pose)

                gt_action = self.action_process(pre_pose)
                self.env.actions = gt_action.clone()
                for env_id in range(self.env.num_envs):
                    self.eps_buffer[env_id].add(pc[env_id], env_state[env_id], gt_action[env_id])

            # update env end flag
            for env_id in range(self.env.num_envs):
                demo_buffer.append(self.eps_buffer[env_id])
            print(f"Episode {eps} Succeeded")
        if self.cfg['env']['collectData']:
            dataset_path = "grasp_cm" + "_" + str(self.cfg["env"]["asset"]["AssetNum"])+ "_eps"+str(self.cfg["task"]["num_episode"]) + "_clock"+str(self.cfg["env"]["clockwise"])
            save_dir = './demo_data/'+ dataset_path
            save_path = save_dir + '/demo_data.zip'
            os.makedirs(save_dir, exist_ok=True)
            demo_buffer.save(save_path)
    
    '''
    manipulation data collect
    '''
    def collect_manip_data(self):
        # move to the handle
        eps_num = self.cfg["task"]["num_episode"]
        policy = self.cfg["task"]["policy"]
        max_step = 20 if policy == "adaptive" else 15
        print("policy_{}--max_step_{}--num_eps_{}".format(policy, max_step, eps_num))
        demo_buffer = Experience()
        succ_cnt = [0] * self.env.num_envs

        for eps in range(eps_num):
            eps_buffer = [Episode_Buffer() for _ in range(self.env.num_envs)]
            done_flag = [False] * self.env.num_envs
            print("eps_{}".format(eps+1))
            self.env.reset()
            pre_pose = self.env.adjust_hand_pose.clone()
            pre_pose[:, 2] += self.env.gripper_length*2
            for i in range(3):
                for j in range(10):
                    self.env.step(pre_pose)
            pre_pose[:, 2] -= self.env.gripper_length + 0.008
            for i in range(3):
                for j in range(10):
                    self.env.step(pre_pose)
            hand_pose = self.env.hand_rigid_body_tensor[:,:7]
            self.env.gripper = True
            for i in range(10):
                self.env.step(hand_pose)
            init_actions = self.action_process(hand_pose)
            self.env.actions = init_actions
            ####################start collect manipulation data############
            step_size = 0.035
            open_size = 0.015
            hand_pose = self.env.hand_rigid_body_tensor[:,:7]
            handle_quat = self.env.part_rigid_body_tensor[:, 3:7]
            rotate_dir = quat_axis(handle_quat, axis=0)
            down_q = torch.stack(self.env.num_envs * [torch.tensor([0, -0.7071068, 0.7071068, 0])]).to(self.env.device).view((self.env.num_envs, 4))
            rotate_dof = self.env.two_dof_tensor[:,0]
            for t in range(max_step):
                cur_p = hand_pose[:,:3]
                pre_p = cur_p.clone()

                for i in range(self.env.num_envs):
                    if policy == "succ":
                        res = self.succ_policy(i)
                    elif policy == "adaptive":
                        res = self.ada_policy(i, t, rotate_dof[i])
                    else:
                        raise NotImplementedError
                    if res == 'z':
                        pre_p[i, 2] -= open_size
                    elif res == 'r':
                        pre_p[i] -= rotate_dir[i] * step_size
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
                for j in range(15):
                    self.env.step(pre_pose)
                self.env.actions = gt_pose
                # update done_flag
                for env_id in range(self.env.num_envs):
                    if (torch.abs(self.env.one_dof_tensor[env_id, 0]) > 0.035).cpu().item() and not done_flag[env_id]:
                        demo_buffer.append(eps_buffer[env_id])
                        done_flag[env_id] = True
                        succ_cnt[env_id] += 1
                        print(f"Env {env_id} Succeeded")
            print(succ_cnt)

        if self.cfg['env']['collectData']:
            dataset_path = "manip_cm_" + self.cfg["task"]["policy"] + "_" + str(self.cfg["env"]["asset"]["AssetNum"]) + "_eps" + str(eps_num)+ "_clock" + str(self.cfg["env"]["clockwise"]) 
            save_dir = './demo_data/'+ dataset_path
            save_path = save_dir + '/demo_data.zip'
            os.makedirs(save_dir, exist_ok=True)
            demo_buffer.save(save_path)   

    
    '''
    test model
    '''
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
        elif -dof < self.env.try_range:
            action = "r"
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