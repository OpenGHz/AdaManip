# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import os
import yaml

from isaacgym import gymapi
from isaacgym import gymutil

import numpy as np
import random
import torch
# import ipdb


def set_np_formatting():
    np.set_printoptions(edgeitems=30, infstr='inf',
                        linewidth=4000, nanstr='nan', precision=2,
                        suppress=False, threshold=10000, formatter=None)


def warn_task_name():
    raise Exception(
        "Unrecognized task!")

def warn_algorithm_name():
    raise Exception(
                "Unrecognized algorithm!\nAlgorithm should be one of: [ppo, happo, hatrpo, mappo]")


def set_seed(seed, torch_deterministic=False):
    
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_deterministic(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

def retrieve_cfg(args):

    log_dir = None
    task_cfg = None

    print("task:",args.task)
    if args.task == "OpenBottle":
        log_dir = os.path.join(args.logdir, "open_bottle/")
        task_cfg = "cfg/open_bottle.yaml"
    elif args.task == "OpenMicroWave":
        log_dir = os.path.join(args.logdir, "open_microwave/")
        task_cfg = "cfg/open_microwave.yaml"
    elif args.task == "OpenDoor":
        log_dir = os.path.join(args.logdir, "open_door/")
        task_cfg = "cfg/open_door.yaml"
    elif args.task == "OpenPen":
        log_dir = os.path.join(args.logdir, "open_pen/")
        task_cfg = "cfg/open_pen.yaml"
    elif args.task == "OpenWindow":
        log_dir = os.path.join(args.logdir, "open_window/")
        task_cfg = "cfg/open_window.yaml"
    elif args.task == "OpenPressureCooker":
        log_dir = os.path.join(args.logdir, "open_pressurecooker/")
        task_cfg = "cfg/open_pressurecooker.yaml"
    elif args.task == "OpenCoffeeMachine":
        log_dir = os.path.join(args.logdir, "open_coffeemachine/")
        task_cfg = "cfg/open_coffeemachine.yaml"
    elif args.task == "OpenLamp":
        log_dir = os.path.join(args.logdir, "open_lamp/")
        task_cfg = "cfg/open_lamp.yaml"
    elif args.task == "OpenSafe":
        log_dir = os.path.join(args.logdir, "open_safe/")
        task_cfg = "cfg/open_safe.yaml"
    elif args.task == "OpenDrawer":
        log_dir = os.path.join(args.logdir, "open_drawer/")
        task_cfg = "cfg/open_drawer.yaml"
    else:
        warn_task_name()
    
    return log_dir, task_cfg


def load_cfg(args):

    with open(os.path.join(os.getcwd(), args.cfg_env), 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    
    cfg["name"] = args.task
    cfg["headless"] = args.headless
    cfg["seed"] = args.seed

    cfg["env"]["asset"]["StartID"] = args.start_id
    
    logdir = args.logdir

    return cfg, logdir


def parse_sim_params(args, cfg):
    # initialize sim
    sim_params = gymapi.SimParams()
    sim_params.dt = 1./60.
    sim_params.num_client_threads = args.slices

    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
        sim_params.flex.shape_collision_margin = 0.01
        sim_params.flex.num_outer_iterations = 4
        sim_params.flex.num_inner_iterations = 10
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.num_threads = 4
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
        sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024

    sim_params.use_gpu_pipeline = args.use_gpu_pipeline
    sim_params.physx.use_gpu = args.use_gpu

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads
    return sim_params


def get_args():
    custom_parameters = [
        {"name": "--headless", "action": "store_true", "default": False,
            "help": "Force display off at all times"},
        {"name": "--task", "type": str, "default": "OpenBottle",
            "help": "Can be OpenBottle, xxx"},
        {"name": "--controller", "type": str, "default": "GTController"},
        {"name": "--manipulation", "type": str, "default": "OpenBottleManipulation"},
        {"name": "--logdir", "type": str, "default": "./logs/"},
        {"name": "--cfg_env", "type": str, "default": "Base"},
        {"name": "--seed", "type": int, "default":-1, "help": "Random seed"},
        {"name": "--start_id", "type": int, "default":0, "help": "Start Index of the loaded assets"},
        {"name": "--runtime_mode", "type": str, "default": "local", "help": "Runtime mode: local, rpyc-server, rpyc-client"},
        {"name": "--rpyc_host", "type": str, "default": "localhost", "help": "RPyC host for server/client mode"},
        {"name": "--rpyc_port", "type": int, "default": 18861, "help": "RPyC port for server/client mode"}]

    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)

    # allignment with examples
    args.device_id = args.compute_device_id
    args.device = args.sim_device_type if args.use_gpu_pipeline else 'cpu'

    logdir, cfg_env = retrieve_cfg(args)

    args.logdir = logdir

    # args.cfg_env = cfg_env

    return args