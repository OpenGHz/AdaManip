import torch
import torch.nn as nn
import numpy as np
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusion_policy.pointnet import PointNetEncoder
from diffusion_policy.seg_pointnet import PointNet2SemSegSSG
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.ema_model import EMAModel
from dataset.dataset import ManipDataset   
from datetime import datetime
from tqdm.auto import tqdm
import ipdb

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

class argument:
    def __init__(self):
        self.ckpt_path = 'checkpoints/ema_nets.pth'
        self.dataset_path = 'demo_data/open_bottle/demo_buffer.zip'
        self.policy_mode = 'diffusion'
        self.pred_horizon = 4
        self.obs_horizon = 2
        self.action_horizon = 1
        self.num_diffusion_iters = 100
        self.flow_sampling_steps = 10
        self.flow_beta_alpha = 1.5
        self.flow_beta_beta = 1.0
        self.flow_tau_cutoff = 0.999
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
    def __init__(self, args):
        self.args = args
        self.policy_mode = getattr(args, 'policy_mode', 'diffusion')
        if self.policy_mode not in ['diffusion', 'flow_matching']:
            raise ValueError(f"Unsupported policy mode: {self.policy_mode}")
        if not hasattr(self.args, 'flow_sampling_steps') or self.args.flow_sampling_steps <= 0:
            self.args.flow_sampling_steps = 10
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

    def build_net(self, args):
        # Initialize Networks
        # vision_encoder = PointNetEncoder(global_feat=True, feature_transform=False, channel=3)
        # self.action_dim, self.obs_dim = self.get_dim(vision_encoder.out_dim)
        vision_encoder = PointNet2SemSegSSG({'input_feat': args.input_feat, 'feat_dim': args.feat_dim})
        self.action_dim = args.action_dim
        self.low_obs_dim = 9 + 9 + 7 + self.args.dof_dim + self.action_dim # no gripper info in prev_action obs
        self.obs_dim = self.low_obs_dim + args.feat_dim
        noise_pred_net = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=self.obs_dim*args.obs_horizon,
            cond_predict_scale=True,
            local_cond_dim=None
        )
        # the final arch has 2 parts
        nets = nn.ModuleDict({
            'vision_encoder': vision_encoder,
            'noise_pred_net': noise_pred_net
        })
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
        if torch.cuda.is_available():
            payload = torch.load(ckpt_path, map_location='cuda')
        else:
            payload = torch.load(ckpt_path, map_location='cpu')

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

    def _encode_obs_condition(self, npcs, npose):
        pcs_features = self.nets['vision_encoder'](npcs)
        obs_features = torch.cat([pcs_features, npose], dim=-1)
        return obs_features.flatten(start_dim=1)

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
        ema = EMAModel(model=self.nets,power=0.75)
        optimizer = torch.optim.AdamW(params=self.nets.parameters(),lr=self.args.lr, weight_decay=self.args.weight_decay)
        lr_scheduler = get_scheduler(
            name='cosine',
            optimizer=optimizer,
            num_warmup_steps=500,
            num_training_steps=len(dataloader) * self.args.num_epochs
        )
        # get current day
        current_day = datetime.now().strftime('%b%d')
        current_time = datetime.now().strftime('%H-%M-%S')

        #current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = self.args.logdir + '/' + current_day + '/' + current_time
        if SummaryWriter is None:
            raise ImportError("tensorboard is required for training but is not installed")
        writer = SummaryWriter(log_dir=log_dir)
        pth_path = log_dir + '/ema_nets.pth'
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
                        obs_cond = self._encode_obs_condition(npcs, npose)

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
                        ema.step(self.nets)

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
                    ema_nets = ema.averaged_model
                    self._save_checkpoint(pth_path, ema_nets.state_dict())
                    print(f"save checkpoint in {epoch_idx} epoch")
        # Weights of the EMA model
        # is used for inference
        ema_nets = ema.averaged_model
        self._save_checkpoint(pth_path, ema_nets.state_dict())
        
    def infer_action_with_seg(self, pcs, env_state):
        npcs, nstate = self._prepare_policy_inputs(pcs, env_state)
        with torch.no_grad():
            obs_cond = self._encode_obs_condition(npcs, nstate)
            naction = self._sample_action_trajectory(obs_cond, nstate.shape[0])
        return naction

    def infer_action(self, pcs, env_state):
        npcs, nstate = self._prepare_policy_inputs(pcs, env_state)
        with torch.no_grad():
            obs_cond = self._encode_obs_condition(npcs, nstate)
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
            obs_cond = self._encode_obs_condition(npcs, npose)

            if self.policy_mode == 'flow_matching':
                primary_loss = self._compute_flow_matching_loss(naction, obs_cond).item()
                primary_label = 'flow loss'
            else:
                primary_loss = self._compute_diffusion_loss(naction, obs_cond).item()
                primary_label = 'noise loss'

            pred_action = self._sample_action_trajectory(obs_cond, B)
            action_loss = nn.functional.mse_loss(pred_action, naction).item()
            
            
            print(primary_label + ": ", primary_loss, "action loss: ", action_loss)

