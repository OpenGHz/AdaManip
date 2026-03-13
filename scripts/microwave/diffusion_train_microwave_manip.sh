python diffusion_train.py --dataset_path ./demo_data/open_microwave_adaptive_10_eps25_clock0.5/demo_data.zip --batch_size 32 --num_epochs 500 --obs_horizon 2 --action_dim 10
# --task_name open_microwave 
# --policy_mode diffusion
# --policy_mode flow_matching
# --resume --ckpt_path ./logs/Mar09/13-49-39/ema_nets.pth
