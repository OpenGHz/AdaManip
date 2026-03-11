# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from envs.open_bottle import OpenBottle
from envs.open_microwave import OpenMicroWave
from envs.open_pen import OpenPen
from envs.open_door import OpenDoor
from envs.open_window import OpenWindow
from envs.open_pressurecooker import OpenPressureCooker
from envs.open_coffeemachine import OpenCoffeeMachine
from envs.open_lamp import OpenLamp
from envs.open_safe import OpenSafe
from ipc.remote_env import RemoteEnv
from utils.config import warn_task_name

def parse_env(args, cfg, sim_params, log_dir):

    if args.runtime_mode == "rpyc-client":
        return RemoteEnv(host=args.rpyc_host, port=args.rpyc_port)

    # create native task and pass custom config
    device_id = args.device_id

    cfg_task = cfg["env"]
    cfg_task["seed"] = cfg["seed"]


    log_dir = log_dir + "_seed{}".format(cfg_task["seed"])

    try:
        task = eval(args.task)(
            cfg=cfg,
            sim_params=sim_params,
            physics_engine=args.physics_engine,
            device_type=args.device,
            device_id=device_id,
            headless=args.headless,
            log_dir=log_dir)
        print(task)
    except NameError as e:
        print(e)
        warn_task_name()
    return task