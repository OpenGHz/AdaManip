from tkinter import NO
from envs.base_env import BaseEnv
from manipulation.base_manipulation import BaseManipulation
from controller.base_controller import BaseController
from logging import Logger
import numpy as np
import torch
import ipdb
from diffusion_policy.diffusion_policy_new import argument
# from diffusion_policy.diffusion_policy_transformer import DiffusionPolicyTran, argument

class ModelController(BaseController) :

    def __init__(self, env : BaseEnv, manipulation : BaseManipulation, cfg : dict, logger : Logger):
        super().__init__(env, manipulation, cfg, logger)
        self.env = env
        self.cfg_model = cfg["model"]
        self.args = argument()
        self.args.dof_dim = self.cfg_model["dof_dim"]
        self.args.ckpt_path = self.cfg_model.get("model_path", self.cfg_model["diffusion_model_path"])
        self.args.obs_horizon = self.cfg_model["obs_horizon"]
        self.args.pred_horizon = self.cfg_model["pred_horizon"]
        self.args.action_horizon = self.cfg_model["action_horizon"]
        self.args.num_diffusion_iters = self.cfg_model["num_diffusion_iters"]
        self.args.policy_mode = self.cfg_model.get("policy_mode", self.args.policy_mode)
        self.args.flow_sampling_steps = self.cfg_model.get("flow_sampling_steps", self.args.flow_sampling_steps)
        self.args.flow_beta_alpha = self.cfg_model.get("flow_beta_alpha", self.args.flow_beta_alpha)
        self.args.flow_beta_beta = self.cfg_model.get("flow_beta_beta", self.args.flow_beta_beta)
        self.args.flow_tau_cutoff = self.cfg_model.get("flow_tau_cutoff", self.args.flow_tau_cutoff)
        self.args.discrete = self.cfg_model["discrete"]
        self.args.input_feat = self.cfg_model["input_feat"]
        self.args.feat_dim = self.cfg_model["feat_dim"]
        self.args.grasp = self.cfg_model["grasp"]
        self.args.action_dim = self.cfg_model["action_dim"]
        self.args.grasp_path = self.cfg_model["grasp_model_path"]
        if self.cfg_model['Transformer']:
            self.args.n_layer = self.cfg_model["n_layer"]
            self.args.n_cond_layers = self.cfg_model["n_cond_layers"]
            self.args.n_head = self.cfg_model["n_head"]
            self.args.n_emb = self.cfg_model["n_emb"]
            self.args.p_drop_emb = self.cfg_model["p_drop_emb"]
            self.args.p_drop_attn = self.cfg_model["p_drop_attn"]
            self.args.causal_attn = self.cfg_model["causal_attn"]
            self.args.time_as_cond = self.cfg_model["time_as_cond"]
            self.args.pred_action_steps_only = self.cfg_model["pred_action_steps_only"]
        self.load_model()
    
    def load_model(self):
        if self.cfg_model['Transformer']:
            from diffusion_policy.diffusion_policy_transformer import DiffusionPolicyTran as DiffusionPolicy
        else:
            from diffusion_policy.diffusion_policy_new import DiffusionPolicy
        if self.args.grasp:
            self.grasp_net = DiffusionPolicy(self.args)
            self.grasp_net.load_checkpoint(self.args.grasp_path)
            self.grasp_net.nets.eval()
        self.manip_net = DiffusionPolicy(self.args)
        self.manip_net.load_checkpoint(self.args.ckpt_path)
        self.manip_net.nets.eval()

    def run(self, eval=True) :
        '''
        Run the controller.
        '''
        if self.args.grasp:
            self.manipulation.diffusion_evaluate(self.grasp_net, self.manip_net)
        else:
            self.manipulation.diffusion_evaluate(self.manip_net)

