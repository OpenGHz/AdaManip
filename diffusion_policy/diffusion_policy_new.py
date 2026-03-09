import torch
import torch.nn as nn
import numpy as np
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from diffusion_policy.pointnet import PointNetEncoder
from diffusion_policy.seg_pointnet import PointNet2SemSegSSG
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from dataset.dataset import ManipDataset   
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import ipdb

class argument:
    def __init__(self):
        self.ckpt_path = 'checkpoints/ema_nets.pth'
        self.dataset_path = 'demo_data/open_bottle/demo_buffer.zip'
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

class DiffusionPolicy:
    def __init__(self, args):
        self.args = args
        self.nets = self.build_net(args)
        self.noise_scheduler = self.get_noise_scheduler(args)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        print("training device:",self.device)
        self.nets.to(self.device)
        print("using new diffusion policy")

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
    
    def load_checkpoint(self, ckpt_path):
        # load checkpoint
        print(f"load checkpoint from {ckpt_path}")
        if torch.cuda.is_available():
            state_dict = torch.load(ckpt_path, map_location='cuda')
        else:
            state_dict = torch.load(ckpt_path, map_location='cpu')
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
        writer = SummaryWriter(log_dir=log_dir)
        pth_path = log_dir + '/ema_nets.pth'

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
                        # ipdb.set_trace()
                        pcs_features = self.nets['vision_encoder'](npcs)
                        # (B,obs_horizon,D)

                        # concatenate vision feature and low-dim obs
                        obs_features = torch.cat([pcs_features, npose], dim=-1)
                        obs_cond = obs_features.flatten(start_dim=1)
                        # (B, obs_horizon * obs_dim)
                        # sample noise to add to actions
                        noise = torch.randn(naction.shape, device=self.device)

                        # sample a diffusion iteration for each data point
                        timesteps = torch.randint(
                            0, self.noise_scheduler.config.num_train_timesteps,
                            (B,), device=self.device
                        ).long()

                        # add noise to the clean actions according to the noise magnitude at each diffusion iteration
                        # (this is the forward diffusion process)
                        noisy_actions = self.noise_scheduler.add_noise(
                            naction, noise, timesteps)
                        # predict the noise residual
                        # ipdb.set_trace()
                        noise_pred = self.nets['noise_pred_net'](
                            noisy_actions, timesteps, global_cond=obs_cond)

                        # L2 loss
                        loss = nn.functional.mse_loss(noise_pred, noise)

                        # optimize
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
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
                writer.add_scalar('Loss/score_loss', np.mean(epoch_loss), epoch_idx)
                #writer.add_scalar('Loss/action_loss', np.mean(epoch_action_loss), epoch_idx)
                if epoch_idx % self.args.save_rate == 0:
                    ema_nets = ema.averaged_model
                    torch.save(ema_nets.state_dict(), pth_path)
                    print(f"save checkpoint in {epoch_idx} epoch")
        # Weights of the EMA model
        # is used for inference
        ema_nets = ema.averaged_model
        torch.save(ema_nets.state_dict(), pth_path)
        
    def infer_action_with_seg(self, pcs, env_state):
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
        with torch.no_grad():
            # pcs features 
            pcs_features = self.nets['vision_encoder'](npcs)
            
            # concat with low-dim observations
            obs_features = torch.cat([pcs_features, nstate], dim=-1)
            B = obs_features.shape[0]
            # if B==1:
            #     obs_features = obs_features.unsqueeze(0)
            # reshape observation to (B,obs_horizon*obs_dim)

            obs_cond = obs_features.flatten(start_dim=1)
            # initialize action from Guassian noise
            noisy_action = torch.randn(
                (B, self.args.pred_horizon, self.action_dim), device=self.device)
            naction = noisy_action
            #print(naction[0][0])
            # init scheduler
            self.noise_scheduler.set_timesteps(self.args.num_diffusion_iters)
            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred = self.nets['noise_pred_net'](
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond
                )

                # inverse diffusion step (remove noise)
                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample
        return naction

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
        with torch.no_grad():
            # pcs features 
            pcs_features = self.nets['vision_encoder'](npcs)
            
            # concat with low-dim observations
            obs_features = torch.cat([pcs_features, nstate], dim=-1)
            B = obs_features.shape[0]
            # if B==1:
            #     obs_features = obs_features.unsqueeze(0)
            # reshape observation to (B,obs_horizon*obs_dim)

            obs_cond = obs_features.flatten(start_dim=1)
            # initialize action from Guassian noise
            noisy_action = torch.randn(
                (B, self.args.pred_horizon, self.action_dim), device=self.device)
            naction = noisy_action
            #print(naction[0][0])
            # init scheduler
            self.noise_scheduler.set_timesteps(self.args.num_diffusion_iters)
            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred = self.nets['noise_pred_net'](
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond
                )

                # inverse diffusion step (remove noise)
                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample
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
        pcs_features,_,_ = self.nets['vision_encoder'](npcs.flatten(end_dim=1))
        pcs_features = pcs_features.reshape(
                *npcs.shape[:2],-1)
        obs_features = torch.cat([pcs_features, npose], dim=-1)
        #obs_features = obs_features.unsqueeze(0)
            # reshape observation to (B,obs_horizon*obs_dim)
        obs_cond = obs_features.flatten(start_dim=1)
        noisy_action = torch.randn(
                (B, self.args.pred_horizon, self.action_dim), device=self.device)
        naction = noisy_action
        # init scheduler
        self.noise_scheduler.set_timesteps(self.args.num_diffusion_iters)

        for k in self.noise_scheduler.timesteps:
            # predict noise
            noise_pred = self.nets['noise_pred_net'](
                sample=naction,
                timestep=k,
                global_cond=obs_cond
            )

            # inverse diffusion step (remove noise)
            naction = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction
            ).prev_sample
        loss = nn.functional.mse_loss(naction, gt_action)

        #infer_action = self.infer_action(npcs, npose)
        return loss
    
    def compute_action_loss(self, dataloader):
        for nbatch in dataloader:
            npcs = nbatch['pcs'][:,:self.args.obs_horizon].to(self.device)
            npose = nbatch['pose'][:,:self.args.obs_horizon,:self.low_obs_dim].to(self.device)
            naction = nbatch['action'].float().to(self.device)
            B = npose.shape[0]
            #print(npcs.shape, npose.shape, naction.shape)
            # encoder vision features
            pcs_features,_,_ = self.nets['vision_encoder'](
                npcs.flatten(end_dim=1))
            pcs_features = pcs_features.reshape(
                *npcs.shape[:2],-1)
            # (B,obs_horizon,D)

            # concatenate vision feature and low-dim obs
            obs_features = torch.cat([pcs_features, npose], dim=-1)
            obs_cond = obs_features.flatten(start_dim=1)
            # (B, obs_horizon * obs_dim)
            

            # sample noise to add to actions
            noise = torch.randn(naction.shape, device=self.device)
            self.noise_scheduler.set_timesteps(self.args.num_diffusion_iters)
            
            # sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (B,), device=self.device
            ).long()

            # add noise to the clean actions according to the noise magnitude at each diffusion iteration
            # (this is the forward diffusion process)
            noisy_actions = self.noise_scheduler.add_noise(
                naction, noise, timesteps)

            # predict the noise residual
            noise_pred = self.nets['noise_pred_net'](
                noisy_actions, timesteps, global_cond=obs_cond)
            noise_loss = nn.functional.mse_loss(noise_pred, noise).item()

            #predict action
            pred_action = torch.randn(naction.shape, device=self.device)
            for t in self.noise_scheduler.timesteps:
                noise_pred = self.nets['noise_pred_net'](
                    pred_action, t, global_cond=obs_cond)
                pred_action = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=t,
                    sample=pred_action
                ).prev_sample
            #print(pred_action.shape, naction.shape)
            action_loss = nn.functional.mse_loss(pred_action, naction).item()
            
            
            print("noise loss: ", noise_loss, "action loss: ", action_loss)

