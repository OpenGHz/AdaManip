from manipulation.base_manipulation import BaseManipulation
from envs.base_env import BaseEnv
from manipulation.utils.transform import *
from logging import Logger
import numpy as np
import random
from dataset.dataset import Experience, Episode_Buffer, obs_wrapper
import os
import collections
import av
import shutil
import time
import json
from pathlib import Path


class Mp4VideoWriter:

    def __init__(self, output_path, width, height, fps, codec, options=None):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.container = av.open(output_path, mode="w")
        self.stream = self.container.add_stream(codec, rate=fps)
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = "yuv420p"
        if options:
            self.stream.options = {key: str(value) for key, value in options.items() if value is not None}

    def write(self, frame):
        video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        for packet in self.stream.encode(video_frame):
            self.container.mux(packet)

    def close(self):
        if self.container is None:
            return
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()
        self.container = None


class EpisodeVideoRecorder:

    def __init__(self, save_dir, cfg, num_envs, width, height, num_fixed_cameras):
        video_cfg = cfg["env"].get("rgbVideo", {})
        self.save_dir = os.path.join(save_dir, "rgb_videos")
        self.camera_type = video_cfg.get("cameraType", "fixed")
        self.fps = int(video_cfg.get("fps", 10))
        self.codec = video_cfg.get("codec", "libx264")
        self.options = {
            "crf": video_cfg.get("crf", 18),
            "preset": video_cfg.get("preset", "fast"),
        }
        configured_camera_ids = video_cfg.get("cameraIds", [])
        if self.camera_type == "fixed":
            self.camera_ids = list(configured_camera_ids) if configured_camera_ids else list(range(num_fixed_cameras))
        else:
            self.camera_ids = [0]
        self.num_envs = num_envs
        self.width = width
        self.height = height
        self.writers = {}
        self.current_episode_dir = None
        self.current_episode_idx = None
        self.temp_dir = os.path.join(self.save_dir, "_tmp")

    def start_episode(self, episode_idx):
        self.close_episode()
        self.current_episode_idx = episode_idx
        self.current_episode_dir = os.path.join(self.temp_dir, f"episode_{episode_idx:04d}")
        if os.path.isdir(self.current_episode_dir):
            shutil.rmtree(self.current_episode_dir)
        for env_id in range(self.num_envs):
            for camera_id in self.camera_ids:
                output_path = os.path.join(
                    self.current_episode_dir,
                    f"env_{env_id:02d}_cam_{camera_id:02d}.mp4",
                )
                self.writers[(env_id, camera_id)] = Mp4VideoWriter(
                    output_path=output_path,
                    width=self.width,
                    height=self.height,
                    fps=self.fps,
                    codec=self.codec,
                    options=self.options,
                )

    def write_frames(self, frames_by_env):
        if not self.writers:
            return
        for env_id, env_frames in enumerate(frames_by_env):
            for frame_idx, frame in enumerate(env_frames):
                camera_id = self.camera_ids[frame_idx]
                self.writers[(env_id, camera_id)].write(frame)

    def close_episode(self):
        for writer in self.writers.values():
            writer.close()
        self.writers = {}

    def finish_episode(self, env_results=None):
        episode_dir = self.current_episode_dir
        episode_idx = self.current_episode_idx
        self.close_episode()

        if episode_dir is None or episode_idx is None:
            return

        if env_results is None:
            final_episode_dir = os.path.join(self.save_dir, f"episode_{episode_idx:04d}")
            if os.path.isdir(final_episode_dir):
                shutil.rmtree(final_episode_dir)
            os.makedirs(os.path.dirname(final_episode_dir), exist_ok=True)
            if os.path.isdir(episode_dir):
                shutil.move(episode_dir, final_episode_dir)
        else:
            for env_id, result in enumerate(env_results):
                result_dir = os.path.join(self.save_dir, result, f"episode_{episode_idx:04d}")
                os.makedirs(result_dir, exist_ok=True)
                for camera_id in self.camera_ids:
                    src_path = os.path.join(episode_dir, f"env_{env_id:02d}_cam_{camera_id:02d}.mp4")
                    if os.path.exists(src_path):
                        shutil.move(src_path, os.path.join(result_dir, os.path.basename(src_path)))
            if os.path.isdir(episode_dir):
                shutil.rmtree(episode_dir)

        self.current_episode_dir = None
        self.current_episode_idx = None

    def discard_episode(self):
        episode_dir = self.current_episode_dir
        self.close_episode()
        if episode_dir is not None and os.path.isdir(episode_dir):
            shutil.rmtree(episode_dir)
        self.current_episode_dir = None
        self.current_episode_idx = None

class OpenMicroWaveManipulation(BaseManipulation) :

    def __init__(self, env : BaseEnv, cfg : dict, logger : Logger) :

        super().__init__(env, cfg, logger)
        self.video_recorder = None

    def _build_microwave_trajectory_label(self, env_id, start_with_pull, expanded_minimal_chains):
        clock_wise = int(self.env.clock_wise[env_id].item())
        if start_with_pull:
            if clock_wise == 0:
                attempt_chain = ["拉门"]
                stage_status = [True]
                minimal_chain = ["拉门"]
            else:
                attempt_chain = ["拉门", "按按钮", "拉门"]
                stage_status = [False, True, True]
                minimal_chain = ["按按钮", "拉门"]
        else:
            attempt_chain = ["按按钮", "拉门"]
            stage_status = [True, True]
            minimal_chain = ["按按钮", "拉门"]

        try:
            minimal_chain_id = expanded_minimal_chains.index(minimal_chain)
        except ValueError as exc:
            raise RuntimeError(f"minimal_chain {minimal_chain} not found in expanded_minimal_chains") from exc

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

    def _build_collect_save_dir(self):
        dataset_path = "open_microwave" + "_" + self.cfg["task"]["policy"] + "_" + str(self.cfg["env"]["asset"]["AssetNum"])+"_eps"+str(self.cfg["task"]["num_episode"])+"_clock"+str(self.cfg["env"]["clockwise"])
        return './demo_data/'+ dataset_path

    def _build_eval_save_dir(self):
        dataset_path = "eval_open_microwave" + "_" + self.cfg["task"]["policy"] + "_" + str(self.cfg["env"]["asset"]["AssetNum"])+"_eps"+str(self.cfg["task"]["num_episode"])+"_clock"+str(self.cfg["env"]["clockwise"])
        run_ts = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        dataset_path = dataset_path + "_" + run_ts
        return './eval_data/'+ dataset_path

    def _prepare_save_dir(self, save_dir, purpose):
        if not os.path.exists(save_dir):
            return

        prompt = f"{purpose} output directory '{save_dir}' already exists. Overwrite it? [y/N]: "
        try:
            answer = input(prompt).strip().lower()
        except EOFError as exc:
            raise RuntimeError(
                f"{purpose} output directory '{save_dir}' already exists and overwrite confirmation could not be read."
            ) from exc

        if answer not in {"y", "yes"}:
            raise RuntimeError(f"Aborted to avoid overwriting existing data at '{save_dir}'.")

        shutil.rmtree(save_dir)

    def _init_video_recorder(self, save_dir):
        if not self.cfg['env'].get('collectRGBVideo', False):
            self.video_recorder = None
            return
        self.video_recorder = EpisodeVideoRecorder(
            save_dir=save_dir,
            cfg=self.cfg,
            num_envs=self.env.num_envs,
            width=self.cfg["env"]["cam"]["width"],
            height=self.cfg["env"]["cam"]["height"],
            num_fixed_cameras=getattr(self.env, "num_cam", 1),
        )

    def _record_video_frame(self):
        if self.video_recorder is None:
            return
        rgb_frames = self.env.collect_rgb_frames(
            camera_type=self.video_recorder.camera_type,
            camera_ids=self.video_recorder.camera_ids,
        )
        self.video_recorder.write_frames(rgb_frames)

    def _load_eval_language_embedding_bank(self, diffusion):
        if not getattr(diffusion.args, 'use_language_conditioning', False):
            return None

        language_path = getattr(diffusion.args, 'language_embedding_dict_path', None)
        if language_path is None:
            ckpt_dir = Path(diffusion.args.ckpt_path).resolve().parent
            language_path = ckpt_dir / 'language_embedding_dict.json'
        else:
            language_path = Path(language_path)

        if not language_path.exists():
            raise FileNotFoundError(
                f'language_embedding_dict.json not found: {language_path}'
            )

        with language_path.open('r', encoding='utf-8') as f:
            payload = json.load(f)

        bank = np.asarray(payload['expanded_minimal_chains'], dtype=np.float32)
        if bank.ndim != 2:
            raise ValueError(f'expanded_minimal_chains must be 2D, got shape {bank.shape}')

        expected_dim = int(getattr(diffusion.args, 'language_input_dim', bank.shape[-1]))
        if bank.shape[-1] != expected_dim:
            raise ValueError(
                f'language embedding dim mismatch: expected {expected_dim}, got {bank.shape[-1]}'
            )

        print(f'loaded language embedding bank: {language_path}, size={bank.shape[0]}')
        return torch.from_numpy(bank).to(diffusion.device)

    def _sample_episode_language_embedding(self, embedding_bank, batch_size):
        if embedding_bank is None:
            return None, None
        bank_size = embedding_bank.shape[0]
        sampled_idx = int(np.random.randint(0, bank_size))
        sampled = embedding_bank[sampled_idx].unsqueeze(0).repeat(batch_size, 1)
        return sampled, sampled_idx

    '''
    test env
    '''
    def test_env(self, pose, eval=False):
        batch_size = pose.shape[0]
        handle_pos = pose[:,:7].clone()
        button_pos = pose[:,7:].clone()
        print(handle_pos)
        print(button_pos)
        '''
        two manipulation choice 1.pull handle open door 2.push button then pull handle open door
        '''
        self.env.reset()
        flag = False
        if flag:
            handle_pos[:, 0] += self.env.gripper_length*2
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
            hand_pose = self.env.hand_rigid_body_tensor[:,:7]
            for i in range(1000):
                self.env.step(hand_pose)
            init_handle_pose = handle_pos.clone()
            init_handle_pose[:, 0] += self.env.gripper_length*2
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
            button_pos[:, 0] += self.env.gripper_length*2 + 0.012
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
            handle_pos[:, 0] += self.env.gripper_length*2
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
    
    def diffusion_evaluate(self, diffusion):
        eps_num = self.cfg["task"]["num_episode"]
        succ_cnt = 0
        succ_rate = []
        language_embedding_bank = self._load_eval_language_embedding_bank(diffusion)
        eval_save_dir = self._build_eval_save_dir()
        self._init_video_recorder(eval_save_dir)
        for eps in range(eps_num):
            print("eps_{}".format(eps+1))
            done_flag = [False] * self.env.num_envs
            episode_results = None
            episode_language_embedding, sampled_language_idx = self._sample_episode_language_embedding(
                language_embedding_bank, self.env.num_envs
            )
            if sampled_language_idx is not None:
                print(f'episode {eps + 1} language embedding id: {sampled_language_idx}')
            self.env.reset(clock_same=False)
            self.env.gripper = torch.zeros((self.env.num_envs,1),device=self.env.device)
            if self.video_recorder is not None:
                self.video_recorder.start_episode(eps)
                self._record_video_frame()
            
            obs = self.env.collect_diff_data()
            pcs, env_state = obs_wrapper(obs)
            pcs_deque = collections.deque([pcs] * diffusion.args.obs_horizon, maxlen=diffusion.args.obs_horizon)
            env_state_deque = collections.deque([env_state] * diffusion.args.obs_horizon, maxlen=diffusion.args.obs_horizon)

            try:
                step = 0
                while step <= 32:
                    action = diffusion.infer_action_with_seg(
                        pcs_deque,
                        env_state_deque,
                        language_embedding=episode_language_embedding,
                    ).detach()
                    action = action[:, :diffusion.args.action_horizon, :]
                    step += diffusion.args.action_horizon
                    for act in range(action.shape[1]):
                        quat = self.rotate_6d_to_quat(action[:, act, 3:9])
                        pre_action = torch.cat([action[:, act, :3], quat], dim=-1)
                        self.env.gripper = (action[:, act, -1] > 0.5).unsqueeze(-1).int()
                        for j in range(15):
                            self.env.step(pre_action)

                        self._record_video_frame()
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

                episode_results = ["success" if success else "failure" for success in done_flag]
            finally:
                if self.video_recorder is not None:
                    if episode_results is None:
                        self.video_recorder.discard_episode()
                    else:
                        self.video_recorder.finish_episode(episode_results)
            cur_rate = succ_cnt/(self.env.num_envs)
            print(f"Eps {eps+1}, current succ rate {cur_rate}")
            succ_rate.append(cur_rate)
            succ_cnt = 0
        print(f"Average Success rate: {np.mean(succ_rate)}")
        print(f"Success rate std: {np.std(succ_rate)}")
        self.video_recorder = None
        return                
    
    def process_data(self, goal_pos):
        obs = self.env.collect_diff_data()
        pc, env_state = obs_wrapper(obs)
        goal_pos = self.action_process(goal_pos)
        # self.env.actions = goal_pos
        if self.env.gripper[0,0].cpu().item() == 1:
            temp = torch.ones((self.env.num_envs,1),device=self.env.device)
        else:
            temp = torch.zeros((self.env.num_envs,1),device=self.env.device)
        action_with_gripper = torch.cat([goal_pos, temp],dim=-1)
        self.env.actions = action_with_gripper
        for env_id in range(self.env.num_envs):
            self.eps_buffer[env_id].add(pc[env_id], env_state[env_id],action_with_gripper[env_id])
            self.append_frame_label(env_id)
        self._record_video_frame()

    def collect_manip_data(self):
        # move to the handle
        eps_num = self.cfg["task"]["num_episode"]
        policy = self.cfg["task"]["policy"]
        template_path, task_spec = self.load_task_language_template("microwave")
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
            print("eps_{}".format(eps+1))
            self.env.reset()
            if self.video_recorder is not None:
                self.video_recorder.start_episode(eps)
            ori_pose = self.env.get_adjust_hand_pose()
            pose = ori_pose.clone()
            handle_pos = pose[:,:7].clone()
            button_pos = pose[:,7:].clone()
            self.env.gripper = torch.zeros((self.env.num_envs,1), device=self.env.device)
            episode_start_with_pull = False
            try:
                if policy == "succ":
                    # succ policy under gt state
                    if self.env.clock_wise[0] == 1:
                        episode_start_with_pull = False
                        # cannot directly open door
                        # push button
                        self.set_current_step(0, "按按钮")
                        button_pos[:, 0] += self.env.gripper_length*2 + 0.012
                        for i in range(2):
                            self.process_data(button_pos)
                            for j in range(15):
                                self.env.step(button_pos)
                        button_pos[:, 0] -= self.env.gripper_length
                        for i in range(2):
                            self.process_data(button_pos)
                            for j in range(15):
                                self.env.step(button_pos)
                        self.env.gripper = torch.ones((self.env.num_envs,1), device=self.env.device)
                        self.process_data(button_pos)
                        for j in range(15):
                            self.env.step(button_pos)
                        button_pos[:, 0] -= 0.03
                        for i in range(2):
                            self.process_data(button_pos)
                            for j in range(15):
                                self.env.step(button_pos)
                        self.env.gripper = torch.zeros((self.env.num_envs,1), device=self.env.device)

                        self.set_current_step(1, "拉门")
                        handle_pos[:, 0] += self.env.gripper_length*2
                        for i in range(2):
                            self.process_data(handle_pos)
                            for j in range(15):
                                self.env.step(handle_pos)
                        handle_pos[:, 0] -= self.env.gripper_length + 0.014
                        for i in range(2):
                            self.process_data(handle_pos)
                            for j in range(15):
                                self.env.step(handle_pos)
                        self.env.gripper = torch.ones((self.env.num_envs,1), device=self.env.device)
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
                        # directly open door
                        self.set_current_step(0, "拉门")
                        handle_pos[:, 0] += self.env.gripper_length*2
                        for i in range(2):
                            self.process_data(handle_pos)
                            for j in range(15):
                                self.env.step(handle_pos)
                        handle_pos[:, 0] -= self.env.gripper_length + 0.014
                        for i in range(2):
                            self.process_data(handle_pos)
                            for j in range(15):
                                self.env.step(handle_pos)
                        self.env.gripper = torch.ones((self.env.num_envs,1), device=self.env.device)
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
                    # ada demo
                    down_q = torch.stack(self.env.num_envs * [torch.tensor([0.7071068, 0.7071068, 0, 0])]).to(self.env.device).view((self.env.num_envs, 4))
                    step_size = 0.045
                    start_with_pull = np.random.rand() < 0.5
                    episode_start_with_pull = start_with_pull

                    if start_with_pull:
                        self.set_current_step(0, "拉门")
                        handle_pos[:, 0] += self.env.gripper_length*2
                        for i in range(2):
                            self.process_data(handle_pos)
                            for j in range(15):
                                self.env.step(handle_pos)
                        handle_pos[:, 0] -= self.env.gripper_length + 0.014
                        for i in range(2):
                            self.process_data(handle_pos)
                            for j in range(15):
                                self.env.step(handle_pos)
                        self.env.gripper = torch.ones((self.env.num_envs,1), device=self.env.device)
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
                        # continue open door
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
                        self.env.gripper = torch.zeros((self.env.num_envs,1), device=self.env.device)
                        keep_pose = self.env.hand_rigid_body_tensor.clone()
                        self.process_data(keep_pose)
                        for i in range(15):
                            self.env.step(keep_pose)
                        
                        keep_pose[:, 0] += self.env.gripper_length
                        for i in range(2):
                            self.process_data(keep_pose)
                            for j in range(15):
                                self.env.step(keep_pose)
                        # push button
                        button_pos[:, 0] += self.env.gripper_length*2 + 0.012
                        for i in range(2):
                            self.process_data(button_pos)
                            for j in range(15):
                                self.env.step(button_pos)
                        button_pos[:, 0] -= self.env.gripper_length
                        for i in range(2):
                            self.process_data(button_pos)
                            for j in range(15):
                                self.env.step(button_pos)
                        self.env.gripper = torch.ones((self.env.num_envs,1), device=self.env.device)
                        self.process_data(button_pos)
                        for j in range(15):
                            self.env.step(button_pos)
                        button_pos[:, 0] -= 0.03
                        for i in range(2):
                            self.process_data(button_pos)
                            for j in range(15):
                                self.env.step(button_pos)
                        self.env.gripper = torch.zeros((self.env.num_envs,1), device=self.env.device)

                        self.set_current_step(pull_step_idx, "拉门")
                        keep_pose = self.env.hand_rigid_body_tensor.clone()
                        keep_pose[:, 0] += self.env.gripper_length
                        self.process_data(keep_pose)
                        for j in range(15):
                            self.env.step(keep_pose)
                        handle_pos = pose[:,:7].clone()
                        handle_pos[:, 0] += self.env.gripper_length*2
                        for i in range(2):
                            self.process_data(handle_pos)
                            for j in range(15):
                                self.env.step(handle_pos)
                        handle_pos[:, 0] -= self.env.gripper_length + 0.014
                        for i in range(2):
                            self.process_data(handle_pos)
                            for j in range(15):
                                self.env.step(handle_pos)
                        self.env.gripper = torch.ones((self.env.num_envs,1), device=self.env.device)
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
                    if (torch.abs(self.env.one_dof_tensor[env_id, 0]) > np.pi/7).cpu().item():
                        demo_buffer.append(self.eps_buffer[env_id])
                        traj_record = self._build_microwave_trajectory_label(
                            env_id=env_id,
                            start_with_pull=episode_start_with_pull,
                            expanded_minimal_chains=expanded_minimal_chains,
                        )
                        traj_record["episode_id"] = saved_episode_id
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

        if self.cfg['env']['collectData']:
            save_path = save_dir + '/demo_data.zip'
            os.makedirs(save_dir, exist_ok=True)
            demo_buffer.save(save_path)
            self.save_language_sidecars(
                save_dir=save_dir,
                template_path=template_path,
                task_name="microwave",
                task_spec=task_spec,
                expanded_minimal_chains=expanded_minimal_chains,
                trajectory_records=trajectory_records,
                frame_records=frame_records,
            )
        self.video_recorder = None
        
    def action_choose(self,t,index,one_motion,two_motion):
        if "r" in self.env.action_chosen[index]:
            if one_motion > 0.0001:
                self.env.action_chosen[index,t] = "z"
                return "z"
            else:
                if two_motion > 0.05:
                    res = random.choice(["z","o"])
                    self.env.action_chosen[index,t] = res
                    return res
                else:
                    if "z" == self.env.action_chosen[index,t-1]:
                        self.env.action_chosen[index,t]= "o"
                        return "o"
                    else:
                        self.env.action_chosen[index,t]= "z"
                        return "z"
        else:
            if one_motion > 0.0001:
                # lift up is successful
                self.env.action_chosen[index,t] = "z"
                return "z"
            else:
                if two_motion > 0.05:
                    # did not lift up, but rotate is successful
                    res = random.choice(["z","o"])
                    self.env.action_chosen[index,t] = res
                    return res
                else:
                    # neither lift up, nor rotate
                    if "o" == self.env.action_chosen[index,t-1]:
                        self.env.action_chosen[index,t]= "z"
                        return "z"
                    else:
                        if t == 0:
                            res = random.choice(["z","o","r"])
                            print("random")
                            self.env.action_chosen[index,t] = res
                            return res
                        elif "z" in self.env.action_chosen[index,t-1]:
                            if "o" in self.env.action_chosen[index]:
                                self.env.action_chosen[index,t]= "o"
                                return "o"
                            else:
                                res = random.choice(["o","r"])
                                self.env.action_chosen[index,t] = res
                                return res
                            
    def pc_normalize(self, pc):
        print("normalize pc", pc.shape)
        print("max_test", torch.max(pc))
        print("min_test", torch.min(pc))
        center = torch.mean(pc, dim=-2, keepdim=True)
        pc = pc - center
        m = torch.max(torch.norm(pc, p=2, dim=-1)).unsqueeze(-1)
        pc = pc / m
        print("max_test", torch.max(torch.norm(pc, p=2, dim=-1)))
        print("min_test", torch.min(torch.norm(pc, p=2, dim=-1)))    
        return pc