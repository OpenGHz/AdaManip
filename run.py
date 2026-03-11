import os
import sys
from ast import arg
import numpy as np
import random
from logging import Logger
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "envs"))
sys.path.append(os.path.join(BASE_DIR, "controller"))

from utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from utils.parse_task import parse_env

from utils.parse import *

def run():
    
    logger = Logger(name=args.task)

    env = parse_env(args, cfg, sim_params, logdir)

    if args.runtime_mode == "rpyc-server":
        from ipc.service import serve_env

        serve_env(
            env,
            host=args.rpyc_host,
            port=args.rpyc_port,
        )
        return

    manipulation = parse_manipulation(args, env, cfg, logger)

    controller = parse_controller(args, env, manipulation, cfg, logger)

    controller.run()


if __name__ == '__main__':

    set_np_formatting()

    args = get_args()

    cfg, logdir = load_cfg(args)

    sim_params = parse_sim_params(args, cfg)

    set_seed(args.seed)

    run()