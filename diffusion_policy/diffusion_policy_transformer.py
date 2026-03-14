import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from typing import Tuple
from einops import reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.seg_pointnet import PointNet2SemSegSSG
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
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
        self.pred_horizon = 4
        self.obs_horizon = 2
        self.action_horizon = 1
        self.num_diffusion_iters = 100
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

class DiffusionPolicyTran:
    def __init__(self, args):
        self.args = args
        self.task_name = getattr(args, 'task_name', None)
        self.task_stage = getattr(args, 'task_stage', 'manip')
        self.nets = self.build_net(args)
        self.noise_scheduler = self.get_noise_scheduler(args)
        self.normalizer = LinearNormalizer()
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        print("training device:",self.device)
        self.nets.to(self.device)
        print("using Transformer diffusion policy")

    def get_optimizer(
            self, 
            transformer_weight_decay: float, 
            obs_encoder_weight_decay: float,
            learning_rate: float, 
            betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:
        optim_groups = self.nets['noise_pred_net'].get_optim_groups(
            weight_decay=transformer_weight_decay)
        optim_groups.append({
            "params": self.nets['vision_encoder'].parameters(),
            "weight_decay": obs_encoder_weight_decay
        })
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def build_net(self, args):
        # Initialize Networks
        vision_encoder = PointNet2SemSegSSG({'input_feat': args.input_feat, 'feat_dim': args.feat_dim})
        self.action_dim, self.obs_dim = self.get_dim(args.feat_dim)
        # transfomer base
        input_dim = self.action_dim
        noise_pred_net = TransformerForDiffusion(
            input_dim = input_dim,
            output_dim = input_dim,
            horizon = args.pred_horizon,
            n_obs_steps = args.obs_horizon,
            cond_dim = self.obs_dim,
            n_layer =  args.n_layer,
            n_head = args.n_head,
            n_emb = args.n_emb,
            p_drop_emb = args.p_drop_emb,
            p_drop_attn = args.p_drop_attn,
            causal_attn = args.causal_attn,
            time_as_cond = args.time_as_cond,
            obs_as_cond = True,
            n_cond_layers = args.n_cond_layers
        )
        # the final arch has 2 parts
        nets = nn.ModuleDict({
            'vision_encoder': vision_encoder,
            'noise_pred_net': noise_pred_net
        })
        self.mask_generator = LowdimMaskGenerator(
            action_dim = self.action_dim,
            obs_dim = 0,
            max_n_obs_steps = args.obs_horizon,
            fix_obs_steps=True,
            action_visible=False
        )
        # self.horizon = args.horizon
        # self.obs_feature_dim = self.obs_dim
        self.n_action_steps = args.pred_horizon
        self.n_obs_steps = args.obs_horizon
        self.obs_as_cond = True
        self.pred_action_steps_only = args.pred_action_steps_only
        return nets

    def get_dim(self, vis_out):
        if self.args.discrete:
            action_dim = 3  # (z, o, r)
        else:
            action_dim = 9  # (translation + quaternion)
        vision_feature_dim = vis_out
        self.low_obs_dim = 9 + 9 + 7 + self.args.dof_dim + action_dim #(qpos + qvel + hand_pose + dof state + prev_action)
        obs_dim = vision_feature_dim + self.low_obs_dim
        return action_dim, obs_dim

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
    
    def load_checkpoint(self, ckpt_path):
        # load checkpoint
        print(f"load checkpoint from {ckpt_path}")
        if torch.cuda.is_available():
            state_dict = torch.load(ckpt_path, map_location='cuda')
        else:
            state_dict = torch.load(ckpt_path, map_location='cpu')
        self.nets.load_state_dict(state_dict)
        self.nets = self.nets.to(self.device)

    def _get_run_log_dir(self):
        current_day = datetime.now().strftime('%b%d')
        current_time = datetime.now().strftime('%H-%M-%S')
        task_name = self.task_name if self.task_name else 'unknown_task'
        task_stage = self.task_stage if self.task_stage else 'manip'
        return os.path.join(self.args.logdir, task_name, task_stage, current_day, current_time)


    def train(self):
        # resume training
        if self.args.resume:
            self.load_checkpoint()
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
        # optimizer = torch.optim.AdamW(params=self.nets.parameters(),lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.optimizer = self.get_optimizer(self.args.transformer_weight_decay,
                                            self.args.obs_encoder_weight_decay,
                                            self.args.learning_rate,
                                            betas=[0.9, 0.95])
        optimizer_to(self.optimizer, self.device)

        lr_scheduler = get_scheduler(
            name='cosine',
            optimizer=self.optimizer,
            num_warmup_steps=500,
            num_training_steps=len(dataloader) * self.args.num_epochs
        )
        log_dir = self._get_run_log_dir()
        if SummaryWriter is None:
            raise ImportError("tensorboard is required for training but is not installed")
        writer = SummaryWriter(log_dir=log_dir)
        pth_path = os.path.join(log_dir, 'ema_nets.pth')
        # configure checkpoint
        # topk_manager = TopKCheckpointManager(
        #     save_dir = os.path.join(log_dir, 'checkpoints'),
        #     monitor_key = "test_mean_score",
        #     mode = "max",
        #     k = 5,
        #     format_str= 'epoch={epoch:04d}-test_mean_score={test_mean_score:.3f}.ckpt'
        # )
        # current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        # log_dir = self.args.logdir + '/' + current_time
        # writer = SummaryWriter(log_dir=log_dir)

        # start training
        with tqdm(range(self.args.num_epochs), desc='Epoch') as tglobal:
            # epoch loop
            for epoch_idx in tglobal:
                epoch_loss = list()
                # batch loop
                with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                    for nbatch in tepoch:
                        # data normalized in dataset
                        # device transfer
                        npcs = nbatch['pcs'][:,:self.args.obs_horizon].to(self.device)
                        npose = nbatch['env_state'][:,:self.args.obs_horizon,:self.low_obs_dim].to(self.device)
                        naction = nbatch['action'][:,:,:self.action_dim].float().to(self.device)
                        #print(npcs.shape, npose.shape, naction.shape)
                        # B = npose.shape[0] 
                        To = self.n_obs_steps

                        # encoder vision features
                        # pcs_features,_,_ = self.nets['vision_encoder'](
                        #     npcs.flatten(end_dim=1))
                        # pcs_features = pcs_features.reshape(
                        #     *npcs.shape[:2],-1)
                        '''
                        seg pointnet
                        '''
                        # ipdb.set_trace()
                        pcs_features = self.nets['vision_encoder'](npcs)
                        # (B,obs_horizon,D)

                        # concatenate vision feature and low-dim obs
                        cond = torch.cat([pcs_features, npose], dim=-1)
                        # obs_cond = obs_features.flatten(start_dim=1)
                        # (B, obs_horizon * obs_dim)
                        # sample noise to add to actions

                        trajectory = naction
                        if self.pred_action_steps_only:
                            start = To - 1
                            end = start + self.n_action_steps
                            trajectory = naction[:,start:end]

                        # generate impainting mask
                        if self.pred_action_steps_only:
                            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
                        else:
                            condition_mask = self.mask_generator(trajectory.shape)

                        # Sample noise that we'll add to the images
                        noise = torch.randn(trajectory.shape, device=trajectory.device)
                        bsz = trajectory.shape[0]
                        # Sample a random timestep for each image
                        timesteps = torch.randint(
                            0, self.noise_scheduler.config.num_train_timesteps, 
                            (bsz,), device=trajectory.device
                        ).long()
                        # Add noise to the clean images according to the noise magnitude at each timestep
                        # (this is the forward diffusion process)
                        noisy_trajectory = self.noise_scheduler.add_noise(
                            trajectory, noise, timesteps)

                        # compute loss mask
                        loss_mask = ~condition_mask

                        # apply conditioning
                        noisy_trajectory[condition_mask] = trajectory[condition_mask]
                        
                        # Predict the noise residual
                        pred = self.nets['noise_pred_net'](noisy_trajectory, timesteps, cond)

                        pred_type = self.noise_scheduler.config.prediction_type 
                        if pred_type == 'epsilon':
                            target = noise
                        elif pred_type == 'sample':
                            target = trajectory
                        else:
                            raise ValueError(f"Unsupported prediction type {pred_type}")

                        loss = F.mse_loss(pred, target, reduction='none')
                        loss = loss * loss_mask.type(loss.dtype).to(self.device)
                        loss = reduce(loss, 'b ... -> b (...)', 'mean')
                        loss = loss.mean()

                        # optimize
                        loss.backward()
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        # step lr scheduler every batch
                        # this is different from standard pytorch behavior
                        lr_scheduler.step()
                        cur_lr = lr_scheduler.get_last_lr()[0]
                        writer.add_scalar('Optimizer/lr', cur_lr, tglobal.n*tepoch.total+tepoch.n)

                        # update Exponential Moving Average of the model weights
                        ema.step(self.nets)
                        
                        # action_loss = nn.functional.mse_loss(pred_action, naction)
                        # logging
                        loss_cpu = loss.item()
                        epoch_loss.append(loss_cpu)
                        #epoch_action_loss.append(action_loss.item())
                        tepoch.set_postfix(loss=loss_cpu)

                # checkpoint
                    
                tglobal.set_postfix(loss=np.mean(epoch_loss))
                writer.add_scalar('Loss/score_loss', np.mean(epoch_loss), epoch_idx)
                #writer.add_scalar('Loss/action_loss', np.mean(epoch_action_loss), epoch_idx)

        # Weights of the EMA model
        # is used for inference
        ema_nets = ema.averaged_model
        pth_path = os.path.join(log_dir, 'ema_nets.pth')
        torch.save(ema_nets.state_dict(), pth_path)

    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            cond=None, generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.nets['noise_pred_net']
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.args.num_diffusion_iters)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output=model_output,
                timestep=t, 
                sample=trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def infer_action(self, pcs, env_state):

        # stack obs
        if len(pcs[0].shape) == 2:
            npcs = torch.stack([p.unsqueeze(0) for p in pcs], axis=1)
            nstate = torch.stack([state.unsqueeze(0) for state in env_state], axis=1)
        else:
            npcs = torch.stack([p for p in pcs], axis=1)
            nstate = torch.stack([state for state in env_state], axis=1)
        # print(npcs.shape, nstate.shape)
        # device transfer
        npcs = npcs.to(self.device, dtype=torch.float32)
        nstate = nstate.to(self.device, dtype=torch.float32)
        cond = None
        cond_data = None
        cond_mask = None

        with torch.no_grad():
            # pcs features 
            pcs_features = self.nets['vision_encoder'](npcs)
            
            # concat with low-dim observations
            obs_features = torch.cat([pcs_features, nstate], dim=-1)
            B = obs_features.shape[0]

            # handle different ways of passing observation
            cond = None
            cond_data = None
            cond_mask = None

            pcs_features = self.nets['vision_encoder'](npcs)
            # (B,obs_horizon,D)

            # concatenate vision feature and low-dim obs
            cond = torch.cat([pcs_features, nstate], dim=-1)

            shape = (B, self.n_action_steps, self.action_dim)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, self.action_dim)
            cond_data = torch.zeros(size=shape, device=self.device, dtype=torch.float32)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

            # run sampling
            action_pred = self.conditional_sample(
                cond_data, 
                cond_mask,
                cond=cond)
                        
            # # unnormalize prediction
            # naction_pred = nsample[...,:Da]
            # action_pred = self.normalizer['action'].unnormalize(naction_pred)
            
            # get action
            if self.pred_action_steps_only:
                action = action_pred
            else:
                start = self.n_obs_steps - 1
                end = start + self.n_action_steps
                action = action_pred[:,start:end]
            
            # result = {
            #     'action': action,
            #     'action_pred': action_pred
            # }
        return action

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        To = self.n_obs_steps

        # handle different ways of passing observation
        cond = None
        trajectory = nactions
        if self.obs_as_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            cond = nobs_features.reshape(batch_size, To, -1)
            if self.pred_action_steps_only:
                start = To - 1
                end = start + self.n_action_steps
                trajectory = nactions[:,start:end]
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            trajectory = torch.cat([nactions, nobs_features], dim=-1).detach()

        # generate impainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)

        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss