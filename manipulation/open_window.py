from manipulation.base_manipulation import BaseManipulation
from envs.base_env import BaseEnv
from manipulation.utils.transform import *
from logging import Logger
import numpy as np
from dataset.dataset import Experience, Episode_Buffer, obs_wrapper
import os
import collections

class OpenWindowManipulation(BaseManipulation) :

    def __init__(self, env : BaseEnv, cfg : dict, logger : Logger) :

        super().__init__(env, cfg, logger)

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
    def diffusion_evaluate(self, grasp_net, diffusion):
        eps_num = self.cfg["task"]["num_eval_episode"]
        policy = self.cfg["task"]["policy"]
        max_step = self.cfg["task"]["max_step"]
        succ_cnt = 0
        succ_rate = []
        print("eval_eps_{},max_step_{},policy_{}".format(eps_num, max_step, policy))
        for eps in range(eps_num):
            self.env.reset()
            done_flag = [False] * self.env.num_envs
            self.diffusion_eval_grasp(grasp_net)

            # Use the latest hand pose after grasp approach as the hold target during gripper settling.
            hand_pose = self.env.hand_rigid_body_tensor[:, :7]
            self.env.gripper = True
            for i in range(10):
                self.env.step(hand_pose)
            init_actions = self.action_process(hand_pose)
            self.env.actions = init_actions
            ###############manipulation policy################
            obs = self.env.collect_diff_data()
            pcs, env_state = obs_wrapper(obs)
            pcs_deque = collections.deque([pcs] * diffusion.args.obs_horizon, maxlen=diffusion.args.obs_horizon)
            env_state_deque = collections.deque([env_state] * diffusion.args.obs_horizon, maxlen=diffusion.args.obs_horizon)
            step = 0
            action_horizon = 1
            while step < max_step:
                pred_poses = diffusion.infer_action_with_seg(pcs_deque, env_state_deque).detach()
                action = pred_poses[:, :action_horizon, :]
                step += action_horizon
                
                for act in range(action.shape[1]):
                    quat = self.rotate_6d_to_quat(action[:, act, 3:])
                    pre_action = torch.cat([action[:, act, :3], quat], dim=-1)
                    self.env.get_obj_dof_property_tensor()

                    for env_id in range(self.env.num_envs):
                        if done_flag[env_id]:
                            pre_action[env_id,:] = hand_pose[env_id,:].clone()
                    for j in range(15):
                        self.env.step(pre_action)
                    
                    self.env.actions = action[:, act, :]
                    obs = self.env.collect_diff_data()
                    pcs, env_state = obs_wrapper(obs)
                        
                    pcs_deque.append(pcs)
                    env_state_deque.append(env_state) 

                for env_id in range(self.env.num_envs):
                    if (torch.abs(self.env.one_dof_tensor[env_id, 0]) > np.pi/6).cpu().item() and not done_flag[env_id]:
                        print(f"Env {env_id} Succeeded")
                        done_flag[env_id] = True
                        succ_cnt += 1 
                
            cur_rate = succ_cnt/(self.env.num_envs)
            print(f"Eps {eps+1}, current succ rate {cur_rate}")
            succ_rate.append(cur_rate)
            succ_cnt = 0
        print(f"Average Success rate: {np.mean(succ_rate)}")
        print(f"Success rate std: {np.std(succ_rate)}")
        return

    '''
    collect grasp data
    '''
    def collect_grasp_data(self):
        eps_num = self.cfg["task"]["num_episode"]
        demo_buffer = Experience()
        for eps in range(eps_num):
            self.eps_buffer = [Episode_Buffer() for _ in range(self.env.num_envs)]
            print("eps_{}".format(eps+1))
            self.env.reset()
            pre_pose = self.env.adjust_hand_pose.clone()
            pre_pose[:,2] += 0.01
            pre_pose[:,0] += self.env.gripper_length*2
            for i in range(3):
                obs = self.env.collect_diff_data()
                pc, env_state = obs_wrapper(obs)

                for j in range(10):
                    self.env.step(pre_pose)

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
                
                gt_action = self.action_process(pre_pose)
                self.env.actions = gt_action.clone()
                for env_id in range(self.env.num_envs):
                    self.eps_buffer[env_id].add(pc[env_id], env_state[env_id], gt_action[env_id])

            # update env end flag
            for env_id in range(self.env.num_envs):
                demo_buffer.append(self.eps_buffer[env_id])
            print(f"Episode {eps} Succeeded")

        if self.cfg['env']['collectData']:
            dataset_path = "grasp_window" + "_" + str(self.cfg["env"]["asset"]["AssetNum"])+"_eps"+str(self.cfg["task"]["num_episode"])+"_clock"+str(self.cfg["env"]["clockwise"])
            save_dir = './demo_data/'+ dataset_path
            save_path = save_dir + '/demo_data.zip'
            os.makedirs(save_dir, exist_ok=True)
            demo_buffer.save(save_path)


    '''
    collect data
    '''
    def collect_manip_data(self):
        eps_num = self.cfg["task"]["num_episode"]
        policy = self.cfg["task"]["policy"]
        print(f"policy: {policy}")
        succ_cnt = [0] * self.env.num_envs
        demo_buffer = Experience()
        for eps in range(eps_num):
            eps_buffer = [Episode_Buffer() for _ in range(self.env.num_envs)]
            done_flag = [False] * self.env.num_envs
            print("eps_{}".format(eps+1))
            self.env.reset()

            pre_pose = self.env.adjust_hand_pose.clone()
            pre_pose[:,2] += 0.01
            pre_pose[:,0] += self.env.gripper_length*2
            for i in range(3):
                for j in range(10):
                    self.env.step(pre_pose)
            pre_pose[:, 0] -= self.env.gripper_length + 0.012
            for i in range(3):
                for j in range(10):
                    self.env.step(pre_pose)

            hand_pose = self.env.hand_rigid_body_tensor[:,:7]

            self.env.gripper = True
            for i in range(10):
                self.env.step(hand_pose)
            init_actions = self.action_process(hand_pose)
            self.env.actions = init_actions
            ####################start collect manipulation data###################
            max_step = 25 if policy == "adaptive" else 20
            rotate_size = 0.04
            open_size = 0.03
            hand_pose = self.env.hand_rigid_body_tensor[:,:7]
            down_q = torch.stack(self.env.num_envs * [torch.tensor([0.0, 1.0, 0, 0])]).to(self.env.device).view((self.env.num_envs, 4))
            rotate_dof = self.env.two_dof_tensor[:,0]
            
            handle_q = self.env.part_rigid_body_tensor[:, 3:7]
            
            for t in range(max_step):
                cur_p = hand_pose[:, :3]
                # cur_q = self.env.hand_rigid_body_tensor[:,3:7]
                pre_p = cur_p.clone()
                # pre_q = cur_q.clone()
                y_rotate_dir = quat_axis(handle_q, axis=0) # y-up
                r_rotate_dir = -y_rotate_dir
                open_dir = quat_axis(handle_q, axis=2)
                for i in range(self.env.num_envs):
                    if policy == "succ":
                        res = self.succ_policy(i)
                    elif policy == "adaptive":
                        res = self.ada_policy(i, t, rotate_dof[i])
                    else:
                        raise NotImplementedError
                    if res == "x":
                        pre_p[i,:] += open_size * open_dir[i].squeeze(0)
                    elif res == "r":
                        pre_p[i,:] += rotate_size * r_rotate_dir[i].squeeze(0)
                    elif res == "y":
                        pre_p[i,:] += rotate_size * y_rotate_dir[i].squeeze(0)
                        
                pred_q = quat_mul(handle_q, down_q)
                pred_pose = torch.cat([pre_p, pred_q], dim=-1).float()
                gt_pose = self.action_process(pred_pose)
                
                for env_id in range(self.env.num_envs):
                    if not done_flag[env_id]:
                        obs = self.env.collect_single_diff_data(env_id)
                        pc, env_state = obs_wrapper(obs)
                        eps_buffer[env_id].add(pc, env_state, gt_pose[env_id])

                for j in range(15):
                    self.env.step(pred_pose)
                
                self.env.actions = gt_pose
                for env_id in range(self.env.num_envs):
                    if (torch.abs(self.env.one_dof_tensor[env_id, 0]) > np.pi/6).cpu().item() and not done_flag[env_id]:
                        demo_buffer.append(eps_buffer[env_id])
                        done_flag[env_id] = True
                        succ_cnt[env_id] += 1
                        print(f"Env {env_id} Succeeded")
            print(succ_cnt)
        if self.cfg['env']['collectData']:
            dataset_path = "manip_window"+"_"+self.cfg["task"]["policy"] + "_" + str(self.cfg["env"]["asset"]["AssetNum"])+"_eps"+str(self.cfg["task"]["num_episode"])+ "_clock"+str(self.cfg["env"]["clockwise"])
            save_dir = './demo_data/'+ dataset_path
            save_path = save_dir + '/demo_data.zip'
            os.makedirs(save_dir, exist_ok=True)
            demo_buffer.save(save_path)
    
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
        clock_wise = self.env.clock_wise[env_id]
        open_flag = self.env.open_bottle_stage[env_id]
        if t == 0:
            action = 'r' if np.random.rand() > 11/20 else 'y'
            self.env.action_chosen[env_id, t] = action
            return action
        elif abs(dof) < self.env.try_range:
            
            if clock_wise:
                self.env.action_chosen[env_id, t] = 'r'
                return 'r'
            else:
                self.env.action_chosen[env_id, t] = 'y'
                return 'y'
        else:
            if self.env.action_chosen[env_id, t-1] == 'x':
                if open_flag:
                    self.env.action_chosen[env_id, t] = 'x'
                    return 'x'
                else:
                    if clock_wise:
                        self.env.action_chosen[env_id, t] = 'r'
                        return 'r'
                    else:
                        self.env.action_chosen[env_id, t] = 'y'
                        return 'y'
            else:
                prob = np.random.rand()
                if prob < 11/20:
                    action = 'x'
                    self.env.action_chosen[env_id, t] = action
                    return action
                else:
                    if clock_wise:
                        self.env.action_chosen[env_id, t] = 'r'
                        return 'r'
                    else:
                        self.env.action_chosen[env_id, t] = 'y'
                        return 'y'
