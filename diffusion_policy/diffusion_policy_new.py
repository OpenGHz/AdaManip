import torch
import torch.nn as nn
import numpy as np
import os
import shutil
from pathlib import Path
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusion_policy.seg_pointnet import PointNet2SemSegSSG
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from dataset.dataset import ManipDataset   
from datetime import datetime
from tqdm.auto import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

class argument:
    def __init__(self):
        self.ckpt_path = 'checkpoints/ema_nets.pth'
        self.dataset_path = 'demo_data/open_bottle/demo_buffer.zip'
        self.task_name = None
        self.task_stage = None
        self.policy_mode = 'diffusion'
        self.pred_horizon = 4
        self.obs_horizon = 2
        self.action_horizon = 1
        self.num_diffusion_iters = 100
        self.flow_sampling_steps = 100
        self.flow_beta_alpha = 1.5
        self.flow_beta_beta = 1.0
        self.flow_tau_cutoff = 0.999
        self.flow_use_ema = False
        self.use_language_conditioning = False
        self.language_input_dim = 512
        self.language_proj_dim = 128
        self.language_embedding_dict_path = None
        self.DDIM = False
        self.discrete = False
        self.dof_dim = 0
        self.num_epochs = 500
        self.load_workers = 8
        self.batch_size = 64
        self.logdir = 'logs'
        self.input_feat = 3
        self.feat_dim = 128
        self.action_dim = 9

class DiffusionPolicy:
    def __init__(self, args: argument):
        self.args = args
        self.task_name = getattr(args, 'task_name', None)
        self.task_stage = getattr(args, 'task_stage', 'manip')
        self.policy_mode = getattr(args, 'policy_mode', 'diffusion')
        self.flow_use_ema = bool(getattr(args, 'flow_use_ema', False))
        self.use_language_conditioning = bool(getattr(args, 'use_language_conditioning', False))
        self.language_input_dim = int(getattr(args, 'language_input_dim', 512))
        self.language_proj_dim = int(getattr(args, 'language_proj_dim', 128))
        if self.policy_mode not in ['diffusion', 'flow_matching']:
            raise ValueError(f"Unsupported policy mode: {self.policy_mode}")
        if not hasattr(self.args, 'flow_sampling_steps') or self.args.flow_sampling_steps <= 0:
            self.args.flow_sampling_steps = 100
        if not hasattr(self.args, 'flow_beta_alpha'):
            self.args.flow_beta_alpha = 1.5
        if not hasattr(self.args, 'flow_beta_beta'):
            self.args.flow_beta_beta = 1.0
        if not hasattr(self.args, 'flow_tau_cutoff'):
            self.args.flow_tau_cutoff = 0.999
        self.args.flow_tau_cutoff = float(min(max(self.args.flow_tau_cutoff, 0.0), 0.999999))
        self.nets = self.build_net(args)
        self.noise_scheduler = self.get_noise_scheduler(args) if self.policy_mode == 'diffusion' else None
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        print("training device:",self.device)
        self.nets.to(self.device)
        print(f"using policy mode: {self.policy_mode}")

        # Diffusion keeps EMA by default; flow matching requires explicit opt-in.
        self.use_ema = self.policy_mode != 'flow_matching' or self.flow_use_ema
        print(f"using EMA: {self.use_ema}")

    def _build_ema_model(self):
        # diffusers API changed from model-based to parameters-based; support both.
        try:
            ema = EMAModel(model=self.nets, power=0.75)
            uses_parameters_api = False
        except TypeError:
            ema = EMAModel(parameters=self.nets.parameters(), power=0.75)
            uses_parameters_api = True
        return ema, uses_parameters_api

    def _step_ema_model(self, ema, uses_parameters_api):
        if uses_parameters_api:
            ema.step(self.nets.parameters())
        else:
            ema.step(self.nets)

    def _get_ema_state_dict(self, ema, uses_parameters_api):
        if not uses_parameters_api and hasattr(ema, "averaged_model"):
            ema_state = ema.averaged_model.state_dict()
        else:
            # For parameters API, materialize EMA parameters into a temporary model.
            temp_model = self.build_net(self.args).eval().cpu()
            temp_model.requires_grad_(False)
            ema.copy_to(temp_model.parameters())
            ema_state = temp_model.state_dict()

        # diffusers parameters-API EMA tracks parameters only; always refresh buffers
        # from the current model state to avoid random/stale BatchNorm statistics.
        param_names = {name for name, _ in self.nets.named_parameters()}
        online_state = self.nets.state_dict()
        for name, value in online_state.items():
            if name not in param_names:
                ema_state[name] = value.detach().cpu().clone()
        return ema_state

    def build_net(self, args):
        # Initialize Networks
        # vision_encoder = PointNetEncoder(global_feat=True, feature_transform=False, channel=3)
        # self.action_dim, self.obs_dim = self.get_dim(vision_encoder.out_dim)
        vision_encoder = PointNet2SemSegSSG({'input_feat': args.input_feat, 'feat_dim': args.feat_dim})
        self.action_dim = args.action_dim
        self.low_obs_dim = 9 + 9 + 7 + self.args.dof_dim + self.action_dim # no gripper info in prev_action obs
        self.obs_dim = self.low_obs_dim + args.feat_dim
        cond_dim = self.obs_dim * args.obs_horizon

        language_proj = None
        if self.use_language_conditioning:
            cond_dim += self.language_proj_dim
            language_proj = nn.Sequential(
                nn.Linear(self.language_input_dim, 256),
                nn.LayerNorm(256),
                nn.Mish(),
                nn.Linear(256, self.language_proj_dim),
            )

        noise_pred_net = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=cond_dim,
            cond_predict_scale=True,
            local_cond_dim=None
        )
        # the final arch has 2 parts
        nets = nn.ModuleDict({
            'vision_encoder': vision_encoder,
            'noise_pred_net': noise_pred_net
        })
        if language_proj is not None:
            nets['language_proj'] = language_proj
        return nets


    def get_noise_scheduler(self, args):
        if args.DDIM:
            print("Using DDIM scheduler")
            noise_scheduler = DDIMScheduler(
                num_train_timesteps=args.num_diffusion_iters,
                # the choise of beta schedule has big impact on performance
                # we found squared cosine works the best
                beta_schedule='squaredcos_cap_v2',
                # clip output to [-1,1] to improve stability
                clip_sample=False,
                # our network predicts noise (instead of denoised action)
                prediction_type='epsilon'
            )
        else:
            print("Using DDPM scheduler")
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=args.num_diffusion_iters,
                # the choise of beta schedule has big impact on performance
                # we found squared cosine works the best
                beta_schedule='squaredcos_cap_v2',
                # clip output to [-1,1] to improve stability
                clip_sample=False,
                # our network predicts noise (instead of denoised action)
                prediction_type='epsilon'
            )
        return noise_scheduler

    def _checkpoint_payload(self, state_dict):
        return {
            'policy_mode': self.policy_mode,
            'state_dict': state_dict,
            'pred_horizon': self.args.pred_horizon,
            'obs_horizon': self.args.obs_horizon,
            'action_dim': self.action_dim,
            'flow_sampling_steps': self.args.flow_sampling_steps,
            'flow_beta_alpha': self.args.flow_beta_alpha,
            'flow_beta_beta': self.args.flow_beta_beta,
            'flow_tau_cutoff': self.args.flow_tau_cutoff,
            'num_diffusion_iters': self.args.num_diffusion_iters,
        }

    def _load_checkpoint_payload(self, ckpt_path):
        map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
        payload = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        if isinstance(payload, dict) and 'state_dict' in payload:
            checkpoint_mode = payload.get('policy_mode')
            state_dict = payload['state_dict']
            is_legacy = False
        elif isinstance(payload, dict):
            checkpoint_mode = 'diffusion'
            state_dict = payload
            is_legacy = True
        else:
            raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")

        if checkpoint_mode is None:
            checkpoint_mode = 'diffusion' if is_legacy else None

        if checkpoint_mode != self.policy_mode:
            raise ValueError(
                f"Checkpoint mode '{checkpoint_mode}' does not match requested policy mode '{self.policy_mode}'."
            )

        return state_dict

    def _save_checkpoint(self, ckpt_path, state_dict):
        torch.save(self._checkpoint_payload(state_dict), ckpt_path)

    def _get_run_log_dir(self):
        current_day = datetime.now().strftime('%b%d')
        current_time = datetime.now().strftime('%H-%M-%S')
        task_name = self.task_name if self.task_name else 'unknown_task'
        task_stage = self.task_stage if self.task_stage else 'manip'
        path_parts = [self.args.logdir, task_name, task_stage]
        path_parts.extend([current_day, current_time])
        return os.path.join(*path_parts)

    def _copy_language_embedding_dict_to_logdir(self, log_dir):
        if not self.use_language_conditioning:
            return

        dataset_paths = self.args.dataset_path
        if isinstance(dataset_paths, (str, Path)):
            dataset_paths = [dataset_paths]

        candidate_sources = []
        language_expanded_sources = []
        for dataset_path in dataset_paths:
            dataset_dir = Path(dataset_path).parent
            embedding_dict_path = dataset_dir / 'language_embedding_dict.json'
            language_expanded_path = dataset_dir / 'language_expanded.json'
            if embedding_dict_path.exists():
                candidate_sources.append(embedding_dict_path)
            if language_expanded_path.exists():
                language_expanded_sources.append(language_expanded_path)

        if not candidate_sources:
            raise ValueError('warning: language conditioning enabled but no language_embedding_dict.json found near dataset path')
        if not language_expanded_sources:
            raise ValueError('warning: language conditioning enabled but no language_expanded.json found near dataset path')

        os.makedirs(log_dir, exist_ok=True)
        target_path = Path(log_dir) / 'language_embedding_dict.json'
        shutil.copy2(candidate_sources[0], target_path)
        print(f'copied language embedding dict to {target_path}')
        target_path = Path(log_dir) / 'language_expanded.json'
        shutil.copy2(language_expanded_sources[0], target_path)
        print(f'copied language expanded dict to {target_path}')

    def _project_language_embedding(self, language_embedding, batch_size, dtype):
        if not self.use_language_conditioning:
            return None
        if language_embedding is None:
            raise ValueError("Language conditioning is enabled but no language_embedding provided") 

        language_embedding = language_embedding.to(self.device, dtype=torch.float32)
        if language_embedding.dim() != 2:
            raise ValueError(
                f"language_embedding must be rank-2 [B, D], got shape {tuple(language_embedding.shape)}"
            )
        if language_embedding.shape[0] != batch_size:
            raise ValueError(
                f"language_embedding batch size mismatch: expected {batch_size}, got {language_embedding.shape[0]}"
            )
        if language_embedding.shape[-1] != self.language_input_dim:
            raise ValueError(
                f"language_embedding dim mismatch: expected {self.language_input_dim}, got {language_embedding.shape[-1]}"
            )

        return self.nets['language_proj'](language_embedding)

    def _encode_obs_condition(self, npcs, npose, language_embedding=None):
        pcs_features = self.nets['vision_encoder'](npcs)
        obs_features = torch.cat([pcs_features, npose], dim=-1)
        obs_flat = obs_features.flatten(start_dim=1)
        lang_proj = self._project_language_embedding(
            language_embedding=language_embedding,
            batch_size=obs_flat.shape[0],
            dtype=obs_flat.dtype,
        )
        if lang_proj is None:
            return obs_flat
        return torch.cat([obs_flat, lang_proj], dim=-1)

    def _prepare_policy_inputs(self, pcs, env_state):
        if len(pcs[0].shape) == 2:
            npcs = torch.stack([p.unsqueeze(0) for p in pcs], axis=1)
            nstate = torch.stack([state.unsqueeze(0) for state in env_state], axis=1)
        else:
            npcs = torch.stack([p for p in pcs], axis=1)
            nstate = torch.stack([state for state in env_state], axis=1)
        npcs = npcs.to(self.device, dtype=torch.float32)
        nstate = nstate.to(self.device, dtype=torch.float32)
        return npcs, nstate

    def _compute_diffusion_loss(self, naction, obs_cond):
        noise = torch.randn(naction.shape, device=self.device)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (naction.shape[0],), device=self.device
        ).long()
        noisy_actions = self.noise_scheduler.add_noise(naction, noise, timesteps)
        noise_pred = self.nets['noise_pred_net'](noisy_actions, timesteps, global_cond=obs_cond)
        return nn.functional.mse_loss(noise_pred, noise)

    def _sample_flow_timesteps(self, batch_size, dtype):
        # Sample low-noise cutoff shifted Beta timesteps as described by pi0.
        beta_dist = torch.distributions.Beta(self.args.flow_beta_alpha, self.args.flow_beta_beta)
        shifted_beta = beta_dist.sample((batch_size,)).to(device=self.device, dtype=dtype)
        return self.args.flow_tau_cutoff * (1.0 - shifted_beta)

    def _compute_flow_matching_loss(self, naction, obs_cond):
        source_actions = torch.randn_like(naction)
        # uniform distribution sampling
        # timesteps = torch.rand((naction.shape[0],), device=self.device, dtype=naction.dtype)
        # beta distribution sampling
        timesteps = self._sample_flow_timesteps(naction.shape[0], naction.dtype)
        time_view = timesteps.view(-1, 1, 1)
        interpolated_actions = (1.0 - time_view) * source_actions + time_view * naction
        # Keep the target aligned with forward Euler rollout from noise to data.
        target_flow = naction - source_actions
        flow_pred = self.nets['noise_pred_net'](interpolated_actions, timesteps, global_cond=obs_cond)
        return nn.functional.mse_loss(flow_pred, target_flow)

    def _sample_diffusion_actions(self, obs_cond, batch_size):
        sampled_actions = torch.randn(
            (batch_size, self.args.pred_horizon, self.action_dim), device=self.device)
        self.noise_scheduler.set_timesteps(self.args.num_diffusion_iters)
        for timestep in self.noise_scheduler.timesteps:
            noise_pred = self.nets['noise_pred_net'](
                sample=sampled_actions,
                timestep=timestep,
                global_cond=obs_cond
            )
            sampled_actions = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=timestep,
                sample=sampled_actions
            ).prev_sample
        return sampled_actions

    def _sample_flow_actions(self, obs_cond, batch_size):
        sampled_actions = torch.randn(
            (batch_size, self.args.pred_horizon, self.action_dim), device=self.device)
        step_count = max(1, self.args.flow_sampling_steps)
        dt = 1.0 / step_count
        for step_idx in range(step_count):
            normalized_time = min((step_idx + 0.5) / step_count, self.args.flow_tau_cutoff)
            timesteps = torch.full(
                (batch_size,), normalized_time, device=self.device, dtype=sampled_actions.dtype)
            flow_pred = self.nets['noise_pred_net'](
                sample=sampled_actions,
                timestep=timesteps,
                global_cond=obs_cond
            )
            sampled_actions = sampled_actions + dt * flow_pred
        return sampled_actions

    def _sample_action_trajectory(self, obs_cond, batch_size):
        if self.policy_mode == 'flow_matching':
            return self._sample_flow_actions(obs_cond, batch_size)
        return self._sample_diffusion_actions(obs_cond, batch_size)
    
    def load_checkpoint(self, ckpt_path):
        # load checkpoint
        print(f"load checkpoint from {ckpt_path}")
        state_dict = self._load_checkpoint_payload(ckpt_path)
        self.nets.load_state_dict(state_dict)
        self.nets = self.nets.to(self.device)

    def train(self):
        if self.args.resume:
            self.load_checkpoint(self.args.ckpt_path)

        # create dataloader
        dataset = ManipDataset(
            dataset_path=self.args.dataset_path,
            pred_horizon=self.args.pred_horizon,
            obs_horizon=self.args.obs_horizon,
            action_horizon=self.args.action_horizon
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.load_workers,
            shuffle=True,
            # accelerate cpu-gpu transfer
            pin_memory=True,
            # don't kill worker process afte each epoch
            persistent_workers=True
        )
        ema = None
        ema_uses_parameters_api = False
        if self.use_ema:
            ema, ema_uses_parameters_api = self._build_ema_model()
        optimizer = torch.optim.AdamW(params=self.nets.parameters(),lr=self.args.lr, weight_decay=self.args.weight_decay)
        lr_scheduler = get_scheduler(
            name='cosine',
            optimizer=optimizer,
            num_warmup_steps=500,
            num_training_steps=len(dataloader) * self.args.num_epochs
        )
        log_dir = self._get_run_log_dir()
        self._copy_language_embedding_dict_to_logdir(log_dir)
        if SummaryWriter is None:
            raise ImportError("tensorboard is required for training but is not installed")
        writer = SummaryWriter(log_dir=log_dir)
        pth_path = os.path.join(log_dir, 'ema_nets.pth')
        loss_name = 'Loss/flow_loss' if self.policy_mode == 'flow_matching' else 'Loss/score_loss'

        # start training
        with tqdm(range(self.args.num_epochs), desc='Epoch') as tglobal:
            # epoch loop
            for epoch_idx in tglobal:
                epoch_loss = list()
                epoch_action_loss = list()
                # batch loop
                with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                    for nbatch in tepoch:
                        # data normalized in dataset
                        # device transfer
                        npcs = nbatch['pcs'][:,:self.args.obs_horizon].to(self.device)
                        assert nbatch['env_state'].shape[-1] == self.low_obs_dim
                        assert nbatch['action'].shape[-1] == self.action_dim
                        npose = nbatch['env_state'][:,:self.args.obs_horizon,:self.low_obs_dim].to(self.device)
                        naction = nbatch['action'][:,:,:self.action_dim].float().to(self.device)
                        #print(npcs.shape, npose.shape, naction.shape)
                        B = npose.shape[0] 

                        '''
                        seg pointnet
                        '''
                        language_embedding = nbatch.get('language_embedding', None)
                        obs_cond = self._encode_obs_condition(npcs, npose, language_embedding=language_embedding)

                        if self.policy_mode == 'flow_matching':
                            loss = self._compute_flow_matching_loss(naction, obs_cond)
                        else:
                            loss = self._compute_diffusion_loss(naction, obs_cond)

                        # optimize
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        # step lr scheduler every batch
                        # this is different from standard pytorch behavior
                        lr_scheduler.step()
                        cur_lr = lr_scheduler.get_last_lr()[0]
                        writer.add_scalar('Optimizer/lr', cur_lr, tglobal.n*tepoch.total+tepoch.n)

                        # update Exponential Moving Average of the model weights
                        if self.use_ema:
                            self._step_ema_model(ema, ema_uses_parameters_api)

                        # logging
                        loss_cpu = loss.item()
                        epoch_loss.append(loss_cpu)
                        #epoch_action_loss.append(action_loss.item())
                        tepoch.set_postfix(loss=loss_cpu)
                tglobal.set_postfix(loss=np.mean(epoch_loss))
                writer.add_scalar(loss_name, np.mean(epoch_loss), epoch_idx)
                writer.add_scalar('Loss/train_loss', np.mean(epoch_loss), epoch_idx)
                #writer.add_scalar('Loss/action_loss', np.mean(epoch_action_loss), epoch_idx)
                if epoch_idx % self.args.save_rate == 0:
                    if self.use_ema:
                        saved_state_dict = self._get_ema_state_dict(ema, ema_uses_parameters_api)
                    else:
                        saved_state_dict = self.nets.state_dict()
                    self._save_checkpoint(pth_path, saved_state_dict)
                    print(f"save checkpoint in {epoch_idx} epoch")
        # Weights of the EMA model
        # is used for inference
        if self.use_ema:
            saved_state_dict = self._get_ema_state_dict(ema, ema_uses_parameters_api)
        else:
            saved_state_dict = self.nets.state_dict()
        self._save_checkpoint(pth_path, saved_state_dict)
        
    def infer_action_with_seg(self, pcs, env_state, language_embedding=None):
        npcs, nstate = self._prepare_policy_inputs(pcs, env_state)
        with torch.no_grad():
            obs_cond = self._encode_obs_condition(npcs, nstate, language_embedding=language_embedding)
            naction = self._sample_action_trajectory(obs_cond, nstate.shape[0])
        return naction

    def infer_action(self, pcs, env_state, language_embedding=None):
        npcs, nstate = self._prepare_policy_inputs(pcs, env_state)
        with torch.no_grad():
            obs_cond = self._encode_obs_condition(npcs, nstate, language_embedding=language_embedding)
            naction = self._sample_action_trajectory(obs_cond, nstate.shape[0])
        return naction

    def compute_loss_single(self, pcs, pose, action):
        # load pcs and pose and action
        self.nets.eval()
        npcs = np.load("demo_data/test/pcs.npy")
        npose = np.load("demo_data/test/pose.npy")
        gt_action = np.load("demo_data/test/action.npy")
        npcs = torch.tensor(npcs[0]).unsqueeze(0).to(self.device)
        npose = torch.tensor(npose[0]).unsqueeze(0).to(self.device)
        gt_action = torch.tensor(gt_action[0]).unsqueeze(0).float().to(self.device)
        B = npose.shape[0]
        # npcs = torch.tensor(pcs[:self.args.obs_horizon]).to(self.device)
        # npose = torch.tensor(pose[:self.args.obs_horizon,:self.low_obs_dim]).to(self.device)
        # gt_action = torch.tensor(action[:self.args.pred_horizon,:self.action_dim]).float().unsqueeze(0).to(self.device)
        obs_cond = self._encode_obs_condition(npcs, npose)
        pred_action = self._sample_action_trajectory(obs_cond, B)
        loss = nn.functional.mse_loss(pred_action, gt_action)

        #infer_action = self.infer_action(npcs, npose)
        return loss
    
    def compute_action_loss(self, dataloader):
        for nbatch in dataloader:
            npcs = nbatch['pcs'][:,:self.args.obs_horizon].to(self.device)
            npose = nbatch['env_state'][:,:self.args.obs_horizon,:self.low_obs_dim].to(self.device)
            naction = nbatch['action'][:,:,:self.action_dim].float().to(self.device)
            B = npose.shape[0]
            language_embedding = nbatch.get('language_embedding', None)
            obs_cond = self._encode_obs_condition(npcs, npose, language_embedding=language_embedding)

            if self.policy_mode == 'flow_matching':
                primary_loss = self._compute_flow_matching_loss(naction, obs_cond).item()
                primary_label = 'flow loss'
            else:
                primary_loss = self._compute_diffusion_loss(naction, obs_cond).item()
                primary_label = 'noise loss'

            pred_action = self._sample_action_trajectory(obs_cond, B)
            action_loss = nn.functional.mse_loss(pred_action, naction).item()
            
            
            print(primary_label + ": ", primary_loss, "action loss: ", action_loss)

