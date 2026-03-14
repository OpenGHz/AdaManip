import argparse
import os


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

    parser.add_argument('--action_dim', type=int, default=9)

    parser.add_argument('--dataset_path', type=str, nargs='+')
    parser.add_argument('--pred_horizon', type=int, default=4)
    parser.add_argument('--obs_horizon', type=int, default=2)
    parser.add_argument('--action_horizon', type=int, default=1)
    parser.add_argument('--dof_dim', type=int, default=0)
    parser.add_argument('--num_diffusion_iters', type=int, default=100)
    parser.add_argument('--DDIM', action='store_true', default=False)
    parser.add_argument('--discrete', action='store_true', default=False)
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
    # seg pointnet para
    parser.add_argument('--input_feat', type=int, default=3)
    parser.add_argument('--feat_dim', type=int, default='128')
    # transformer para
    # parser.add_argument('--horizon', type=int, default=10)
    parser.add_argument('--n_layer', type=int, default=8)
    parser.add_argument('--n_cond_layers', type=int, default=0) # >0: use transformer encoder for cond, otherwise use MLP

    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--n_emb', type=int, default=256)
    parser.add_argument('--p_drop_emb', type=float, default=0.0)
    parser.add_argument('--p_drop_attn', type=float, default=0.3)
    
    parser.add_argument('--causal_attn', action='store_true', default=True)
    parser.add_argument('--time_as_cond', action='store_true', default=True)  # if false, use BERT like encoder only arch, time as input
    parser.add_argument('--pred_action_steps_only', action='store_true', default=False)

    # optimizer
    parser.add_argument('--transformer_weight_decay', type=float, default=1.0e-3)
    parser.add_argument('--obs_encoder_weight_decay', type=float, default=1.0e-6)
    parser.add_argument('--learning_rate', type=float, default=1.0e-4)

    # control
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--ckpt_path', type=str, default='checkpoints/ema_nets.pth')
    

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    inferred_task_name, inferred_stage = infer_task_info(args.dataset_path)
    if args.task_name is None:
        args.task_name = inferred_task_name
    if args.task_stage is None:
        args.task_stage = inferred_stage
    if args.task_name is None:
        args.task_name = 'unknown_task'
    if args.task_stage is None:
        args.task_stage = 'manip'
    from diffusion_policy.diffusion_policy_new import DiffusionPolicy
    policy = DiffusionPolicy(args)
    # from diffusion_policy.diffusion_policy_transformer import DiffusionPolicyTran
    # policy = DiffusionPolicyTran(args)
    
    policy.train()