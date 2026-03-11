# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import importlib

from utils.config import warn_task_name


TASK_REGISTRY = {
    "OpenBottle": ("envs.open_bottle", "OpenBottle"),
    "OpenMicroWave": ("envs.open_microwave", "OpenMicroWave"),
    "OpenPen": ("envs.open_pen", "OpenPen"),
    "OpenDoor": ("envs.open_door", "OpenDoor"),
    "OpenWindow": ("envs.open_window", "OpenWindow"),
    "OpenPressureCooker": ("envs.open_pressurecooker", "OpenPressureCooker"),
    "OpenCoffeeMachine": ("envs.open_coffeemachine", "OpenCoffeeMachine"),
    "OpenLamp": ("envs.open_lamp", "OpenLamp"),
    "OpenSafe": ("envs.open_safe", "OpenSafe"),
}


def _load_task_class(task_name):
    try:
        module_name, class_name = TASK_REGISTRY[task_name]
    except KeyError:
        warn_task_name()

    module = importlib.import_module(module_name)
    return getattr(module, class_name)

def parse_env(args, cfg, sim_params, log_dir):

    if args.runtime_mode == "rpyc-client":
        from ipc.remote_env import RemoteEnv

        return RemoteEnv(host=args.rpyc_host, port=args.rpyc_port)

    # create native task and pass custom config
    device_id = args.device_id

    cfg_task = cfg["env"]
    cfg_task["seed"] = cfg["seed"]


    log_dir = log_dir + "_seed{}".format(cfg_task["seed"])

    task_class = _load_task_class(args.task)
    task = task_class(
        cfg=cfg,
        sim_params=sim_params,
        physics_engine=args.physics_engine,
        device_type=args.device,
        device_id=device_id,
        headless=args.headless,
        log_dir=log_dir)
    print(task)
    return task