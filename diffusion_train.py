import argparse
import os
import sys

import yaml


MODEL_CONFIG_KEYS = {
    'policy_mode',
    'pred_horizon',
    'obs_horizon',
    'action_horizon',
    'dof_dim',
    'num_diffusion_iters',
    'DDIM',
    'discrete',
    'input_feat',
    'feat_dim',
    'n_layer',
    'n_cond_layers',
    'n_head',
    'n_emb',
    'p_drop_emb',
    'p_drop_attn',
    'causal_attn',
    'time_as_cond',
    'pred_action_steps_only',
    'action_dim',
    'flow_sampling_steps',
    'flow_beta_alpha',
    'flow_beta_beta',
    'flow_tau_cutoff',
    'flow_use_ema',
    'use_language_conditioning',
    'language_input_dim',
    'language_proj_dim',
    'language_embedding_dict_path',
}

ROLE_KEYS = {'grasp', 'manip'}


def load_train_cfg(cfg_env):
    cfg_env_path = os.path.abspath(os.path.join(os.getcwd(), cfg_env))
    with open(cfg_env_path, 'r', encoding='utf-8') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    return cfg or {}, cfg_env_path


def provided_arg_dests(parser, argv):
    provided = set()
    for token in argv:
        for action in parser._actions:
            for option in action.option_strings:
                if token == option or token.startswith(option + '='):
                    provided.add(action.dest)
    return provided


def select_train_cfg(train_cfg, task_stage):
    if not isinstance(train_cfg, dict):
        return {}

    selected = {key: value for key, value in train_cfg.items() if key not in ROLE_KEYS}
    if task_stage in ROLE_KEYS and isinstance(train_cfg.get(task_stage), dict):
        selected.update(train_cfg[task_stage])
    elif task_stage is None:
        role_sections = [key for key in ROLE_KEYS if isinstance(train_cfg.get(key), dict)]
        if len(role_sections) == 1:
            role = role_sections[0]
            selected.update(train_cfg[role])
            selected.setdefault('task_stage', role)
    return selected


def config_defaults(cfg, task_stage):
    defaults = {}
    model_cfg = cfg.get('model') or {}
    for key in MODEL_CONFIG_KEYS:
        if key in model_cfg:
            defaults[key] = model_cfg[key]

    env_cfg = cfg.get('env') or {}
    if 'action_dim' not in defaults and 'numActions' in env_cfg:
        defaults['action_dim'] = env_cfg['numActions']

    defaults.update(select_train_cfg(cfg.get('train'), task_stage))
    return defaults


def apply_config_defaults(args, parser, cfg, provided):
    valid_dests = {action.dest for action in parser._actions}
    for dest, value in config_defaults(cfg, args.task_stage).items():
        if dest in valid_dests and dest not in provided:
            setattr(args, dest, value)


def normalize_args(args):
    if isinstance(args.dataset_path, str):
        args.dataset_path = [args.dataset_path]
    return args


def print_resolved_args(args):
    keys = [
        'cfg_env',
        'dataset_path',
        'task_name',
        'task_stage',
        'policy_mode',
        'obs_horizon',
        'action_horizon',
        'pred_horizon',
        'action_dim',
        'num_diffusion_iters',
        'flow_sampling_steps',
        'flow_beta_alpha',
        'flow_beta_beta',
        'flow_tau_cutoff',
        'flow_use_ema',
        'use_language_conditioning',
        'language_input_dim',
        'language_proj_dim',
        'batch_size',
        'num_epochs',
        'load_workers',
        'save_rate',
        'lr',
        'weight_decay',
        'logdir',
    ]
    for key in keys:
        print(f'{key}: {getattr(args, key, None)}')


def infer_task_info(dataset_path_list):
    if not dataset_path_list:
        return None, None

    first_path = os.path.normpath(dataset_path_list[0])
    base_name = os.path.basename(first_path)
    if base_name in {'demo_data.zip', 'demo_buffer.zip'}:
        base_name = os.path.basename(os.path.dirname(first_path))

    if not base_name:
        return None, None

    name = os.path.splitext(base_name)[0]
    tokens = [tok for tok in name.split('_') if tok]
    if not tokens:
        return None, None

    stage = None
    task_name = None

    if tokens[0] in {'grasp', 'manip'}:
        stage = tokens[0]
        if len(tokens) > 1:
            task_name = tokens[1]
    elif tokens[0] == 'open':
        stage = 'manip'
        if len(tokens) > 1:
            task_name = tokens[1]
    else:
        if 'grasp' in tokens:
            grasp_idx = tokens.index('grasp')
            stage = 'grasp'
            if grasp_idx + 1 < len(tokens):
                task_name = tokens[grasp_idx + 1]
        elif 'manip' in tokens:
            manip_idx = tokens.index('manip')
            stage = 'manip'
            if manip_idx + 1 < len(tokens):
                task_name = tokens[manip_idx + 1]

    if task_name is None:
        task_name = tokens[0]
    if stage is None:
        stage = 'manip'

    return task_name, stage

def get_args():
    # use parser to get args
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg_env', type=str, default=None)
    parser.add_argument('--action_dim', type=int, default=9)

    parser.add_argument('--dataset_path', type=str, nargs='+', default=None)
    parser.add_argument('--pred_horizon', type=int, default=4)
    parser.add_argument('--obs_horizon', type=int, default=2)
    parser.add_argument('--action_horizon', type=int, default=1)
    parser.add_argument('--dof_dim', type=int, default=0)
    parser.add_argument('--num_diffusion_iters', type=int, default=100)
    parser.add_argument('--DDIM', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--discrete', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--load_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--save_rate', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--logdir', type=str, default='logs')
    parser.add_argument('--task_name', type=str, default=None)
    parser.add_argument('--task_stage', type=str, default=None, choices=['grasp', 'manip'])
    parser.add_argument('--policy_mode', type=str, default='diffusion', choices=['diffusion', 'flow_matching'])
    parser.add_argument('--flow_sampling_steps', type=int, default=10)
    parser.add_argument('--flow_beta_alpha', type=float, default=1.5)
    parser.add_argument('--flow_beta_beta', type=float, default=1.0)
    parser.add_argument('--flow_tau_cutoff', type=float, default=0.999)
    parser.add_argument('--flow_use_ema', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--use_language_conditioning', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--language_input_dim', type=int, default=512)
    parser.add_argument('--language_proj_dim', type=int, default=128)
    parser.add_argument('--language_embedding_dict_path', type=str, default=None)
    # seg pointnet para
    parser.add_argument('--input_feat', type=int, default=3)
    parser.add_argument('--feat_dim', type=int, default=128)
    # transformer para
    # parser.add_argument('--horizon', type=int, default=10)
    parser.add_argument('--n_layer', type=int, default=8)
    parser.add_argument('--n_cond_layers', type=int, default=0) # >0: use transformer encoder for cond, otherwise use MLP

    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--n_emb', type=int, default=256)
    parser.add_argument('--p_drop_emb', type=float, default=0.0)
    parser.add_argument('--p_drop_attn', type=float, default=0.3)
    
    parser.add_argument('--causal_attn', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--time_as_cond', action=argparse.BooleanOptionalAction, default=True)  # if false, use BERT like encoder only arch, time as input
    parser.add_argument('--pred_action_steps_only', action=argparse.BooleanOptionalAction, default=False)

    # optimizer
    parser.add_argument('--transformer_weight_decay', type=float, default=1.0e-3)
    parser.add_argument('--obs_encoder_weight_decay', type=float, default=1.0e-6)
    parser.add_argument('--learning_rate', type=float, default=1.0e-4)

    # control
    parser.add_argument('--resume', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--ckpt_path', type=str, default='checkpoints/ema_nets.pth')
    parser.add_argument('--dry_run', action='store_true', default=False)
    

    args = parser.parse_args()
    if args.cfg_env:
        cfg, cfg_env_path = load_train_cfg(args.cfg_env)
        args.cfg_env_path = cfg_env_path
        apply_config_defaults(args, parser, cfg, provided_arg_dests(parser, sys.argv[1:]))

    return normalize_args(args)

if __name__ == '__main__':
    args = get_args()
    if not args.dataset_path:
        raise ValueError('dataset_path is required. Set train.<role>.dataset_path in --cfg_env or pass --dataset_path.')
    inferred_task_name, inferred_stage = infer_task_info(args.dataset_path)
    if args.task_name is None:
        args.task_name = inferred_task_name
    if args.task_stage is None:
        args.task_stage = inferred_stage
    if args.task_name is None:
        args.task_name = 'unknown_task'
    if args.task_stage is None:
        args.task_stage = 'manip'
    if args.dry_run:
        print_resolved_args(args)
        sys.exit(0)
    from diffusion_policy.diffusion_policy_new import DiffusionPolicy
    policy = DiffusionPolicy(args)
    # from diffusion_policy.diffusion_policy_transformer import DiffusionPolicyTran
    # policy = DiffusionPolicyTran(args)
    
    policy.train()