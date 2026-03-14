from manipulation.base_manipulation import BaseManipulation
from envs.base_env import BaseEnv
from manipulation.utils.transform import *
from logging import Logger
import numpy as np
from dataset.dataset import Experience, Episode_Buffer, obs_wrapper
import os
import collections
# from utils.o3dviewer import torch2o3d

class OpenSafeManipulation(BaseManipulation) :

    def __init__(self, env : BaseEnv, cfg : dict, logger : Logger) :

        super().__init__(env, cfg, logger)

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
    
    def diffusion_evaluate(self, diffusion):
        eps_num = self.cfg["task"]["num_episode"]
        succ_cnt = 0
        succ_rate = []
        for eps in range(eps_num):
            print("eps_{}".format(eps+1))
            done_flag = [False] * self.env.num_envs
            self.env.reset(clock_same=False)    
            self.env.gripper = torch.zeros((self.env.num_envs,1),device=self.env.device)        
            obs = self.env.collect_diff_data()
            pcs, env_state = obs_wrapper(obs)
            pcs_deque = collections.deque([pcs] * diffusion.args.obs_horizon, maxlen=diffusion.args.obs_horizon)
            env_state_deque = collections.deque([env_state] * diffusion.args.obs_horizon, maxlen=diffusion.args.obs_horizon)
            hand_pose = self.env.hand_rigid_body_tensor[:,:7]
            step = 0
            while step <= 50:
                action = diffusion.infer_action_with_seg(pcs_deque, env_state_deque).detach()
                action = action[:, :diffusion.args.action_horizon, :]
                step += diffusion.args.action_horizon
                for act in range(action.shape[1]):
                    quat = self.rotate_6d_to_quat(action[:, act, 3:9])
                    pre_action = torch.cat([action[:, act, :3], quat], dim=-1)
                    self.env.gripper = (action[:, act, -1] > 0.5).unsqueeze(-1).int()
                    for env_id in range(self.env.num_envs):
                        if done_flag[env_id]:
                            pre_action[env_id, :] = hand_pose[env_id, :]
                    for j in range(15):
                        self.env.step(pre_action)
                    self.env.actions = action[:, act, :]
                    obs = self.env.collect_diff_data()
                    pcs, env_state = obs_wrapper(obs)
                    pcs_deque.append(pcs)
                    env_state_deque.append(env_state)
                
                for env_id in range(self.env.num_envs):
                    if (torch.abs(self.env.one_dof_tensor[env_id, 0]) > np.pi/7).cpu().item() and not done_flag[env_id]:
                        done_flag[env_id] = True
                        succ_cnt += 1
                        print(f"Env {env_id} Succeeded") 
            cur_rate = succ_cnt/(self.env.num_envs)
            print(f"Eps {eps+1}, current succ rate {cur_rate}")
            succ_rate.append(cur_rate)
            succ_cnt = 0
        print(f"Average Success rate: {np.mean(succ_rate)}")
        print(f"Success rate std: {np.std(succ_rate)}")
    
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
        

    def collect_manip_data(self):
        eps_num = self.cfg["task"]["num_episode"]
        policy = self.cfg["task"]["policy"]
        rot_quat = torch.tensor([ 0, 0, -0.258819, 0.9659258], device=self.env.device) 
        s_rot_quat = torch.tensor([ 0, 0, 0.258819, 0.9659258], device=self.env.device) 

        all_demo_buffer = Experience() # Save the continuous action trajectory in the whole episode
        for eps in range(eps_num):
            self.all_eps_buffer = [Episode_Buffer() for _ in range(self.env.num_envs)]
            print("eps_{}".format(eps+1))
            self.env.reset()
            # print(self.env.clock_wise)
            pose = self.env.get_adjust_hand_pose().clone()
            handle_pos = pose[:,:7]
            knob_pos = pose[:,7:]
            knob_pos[:,0] -= 0.002
            if policy == "succ":
                if self.env.clock_wise[0]: # locked
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

        if self.cfg['env']['collectData']:
            dataset_path = "open_safe" + "_" + self.cfg["task"]["policy"] + "_" + str(self.cfg["env"]["asset"]["AssetNum"])+"_eps"+str(self.cfg["task"]["num_episode"])+"_clock"+str(self.cfg["env"]["clockwise"])
            save_dir = './demo_data/'+ dataset_path 
            save_path = save_dir + '/demo_data.zip'            
            os.makedirs(save_dir, exist_ok=True)
            all_demo_buffer.save(save_path)
            print("Demo saved")


